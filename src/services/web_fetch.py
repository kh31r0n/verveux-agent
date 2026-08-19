"""SSRF-hardened HTML fetching for agents that follow tenant-supplied URLs.

Why this module exists
----------------------
``prospecting_nodes._fetch_page_text`` fetches URLs that came from a Serper SERP
response — attacker influence there is indirect. The **enrichment** agent
(sherlock) fetches ``Contact.website``, a field a tenant user (or the prospecting
agent) typed in. That is directly attacker-controllable input handed to an
outbound HTTP client running inside our VPC, i.e. a textbook SSRF sink. Every
guard below exists because of that difference:

* scheme allow-list (``http``/``https`` only);
* DNS resolved ONCE, every returned address validated against an explicit
  network block-list, then the connection is **pinned** to a validated IP so the
  name cannot re-resolve to a private address between check and connect
  (DNS rebinding);
* redirects are followed manually, and every hop is re-validated from scratch;
* the response body is capped in **wire** bytes with compression disabled, so a
  gzip bomb cannot expand past the cap before we notice;
* one total wall-clock budget for the whole operation, because httpx timeouts are
  per-operation and a server trickling bytes can otherwise stream forever;
* ``<script>``/``<style>`` bodies are dropped before text extraction.

Residual risks (NOT closed here, stated so nobody assumes otherwise)
-------------------------------------------------------------------
1. A hostname that legitimately resolves to a **public** IP but fronts an
   internal service (a public ALB routing into the VPC) passes every check here.
   The only real mitigation is an egress-restricted security group.
2. Open redirectors on an otherwise-allowed host.
3. We do not consult ``robots.txt``.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import parse_qs, urlsplit

import httpx
import structlog
import tldextract

logger = structlog.get_logger(__name__)

# Site-ownership boundary for the crawl. See ``_site_key`` for why this is a
# separate extractor from ``agents.dedup._EXTRACT``.
_SITE_EXTRACT = tldextract.TLDExtract(
    # Use tldextract's bundled PSL snapshot only.  ``cache_dir=None`` is
    # important in containers that run with HOME=/nonexistent: otherwise the
    # library still attempts to cache that bundled data under HOME on first use.
    suffix_list_urls=(),
    cache_dir=None,
    include_psl_private_domains=True,
)

# ── Policy constants ─────────────────────────────────────────────────────────

ALLOWED_SCHEMES = ("http", "https")

# Content types we are willing to read. A missing/unknown type fails CLOSED:
# we are looking for a company's contact page, not probing arbitrary services.
ALLOWED_CONTENT_TYPES = frozenset(
    {"text/html", "application/xhtml+xml", "text/plain"}
)

USER_AGENT = "VerveuxEnrichment/1.0 (+https://verveux.com/bot)"

# Explicit network block-list rather than ``ip.is_private``/``is_global``.
# Those properties changed semantics for 0.0.0.0/8, 100.64.0.0/10, 192.0.0.0/24
# and IPv4-mapped IPv6 across CPython patch releases (gh-113171, fixed in 3.11.9
# / 3.12.4). This project only requires >=3.11, so the runtime may predate the
# fix; an explicit list behaves identically on every version.
_BLOCKED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        # ── IPv4 ──
        "0.0.0.0/8",  # "this host on this network"
        "10.0.0.0/8",  # RFC1918
        "100.64.0.0/10",  # CGNAT
        "127.0.0.0/8",  # loopback
        "169.254.0.0/16",  # link-local, incl. 169.254.169.254 cloud metadata
        "172.16.0.0/12",  # RFC1918
        "192.0.0.0/24",  # IETF protocol assignments
        "192.0.2.0/24",  # TEST-NET-1
        "192.88.99.0/24",  # deprecated 6to4 relay anycast
        "192.168.0.0/16",  # RFC1918
        "198.18.0.0/15",  # benchmarking
        "198.51.100.0/24",  # TEST-NET-2
        "203.0.113.0/24",  # TEST-NET-3
        "224.0.0.0/4",  # multicast
        "240.0.0.0/4",  # reserved
        "255.255.255.255/32",  # broadcast
        # ── IPv6 ──
        "::/128",  # unspecified
        "::1/128",  # loopback
        "::ffff:0:0/96",  # IPv4-mapped (also unwrapped explicitly below)
        "64:ff9b::/96",  # NAT64
        "100::/64",  # discard-only
        "2001::/32",  # Teredo
        "2001:db8::/32",  # documentation
        "2002::/16",  # 6to4
        "fc00::/7",  # unique-local
        "fe80::/10",  # link-local
        "ff00::/8",  # multicast
    )
)

# GCP/AWS metadata also answer on these names; blocking the IPs covers the
# resolved address, but reject the well-known names early for a clearer signal.
_BLOCKED_HOSTNAMES = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)


class FetchBlocked(Exception):
    """A URL was refused by policy. ``reason`` is a stable metric label."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass
class FetchedPage:
    url: str
    text: str
    links: list[str] = field(default_factory=list)
    # Contact signals found deterministically in the HTML (before its visible
    # text is capped). These are evidence, not trusted phone numbers: the
    # backend still normalizes and validates every value as E.164.
    contact_signals: list[dict[str, str]] = field(default_factory=list)
    bytes_read: int = 0


# ── HTML extraction ──────────────────────────────────────────────────────────


_PHONE_TOKEN_RE = re.compile(r"(?<!\d)\+?\d[\d\s().-]{5,}\d(?!\d)")
_PHONE_LABEL_RE = re.compile(
    r"(?i)(tel[eé]fono|tel\.?|phone|celular|m[oó]vil|mobile|recepci[oó]n|"
    r"admisiones|cartera|tesorer[ií]a|fijo|contacto)\s*:?\s*$"
)
_WHATSAPP_HOSTS = frozenset({"wa.me", "api.whatsapp.com", "web.whatsapp.com"})


class _TextAndLinkExtractor(HTMLParser):
    """Collect visible text, crawl links, and deterministic contact signals.

    Uses stdlib ``HTMLParser`` rather than a regex on purpose. A
    ``re.sub(r"<[^>]+>", " ", html)`` strip removes *tags* but keeps the
    **contents** of ``<script>``/``<style>``, and on a modern site those live in
    ``<head>`` and run to tens of kilobytes of minified JavaScript. With a
    character budget applied afterwards, the budget is spent on JS and the footer
    — where the phone number actually is — never reaches the model. The same
    regex also mis-handles ``<!-- a > b -->`` and ``title="a>b"``, and does not
    decode entities, so ``&#43;57…`` would survive unusable.

    ``convert_charrefs`` (default True) handles entity decoding for us.
    """

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self.links: list[str] = []
        self._contact_signals: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        attr_map = {name.lower(): value for name, value in attrs if value}
        label = (
            attr_map.get("aria-label")
            or attr_map.get("title")
            or ""
        ).strip()
        if tag == "a":
            href = attr_map.get("href")
            if href:
                self.links.append(href)
                self._extract_href_contact(href, label)

        # JoinChat stores its number as JSON in data-settings rather than in
        # visible markup. Only inspect known contact attributes — never every
        # arbitrary data-* attribute — so page metadata cannot masquerade as a
        # phone number.
        settings_raw = attr_map.get("data-settings")
        if settings_raw:
            try:
                settings = json.loads(settings_raw)
            except (TypeError, json.JSONDecodeError):
                settings = {}
            if isinstance(settings, dict):
                telephone = settings.get("telephone")
                if isinstance(telephone, (str, int)):
                    self._add_signal(
                        str(telephone),
                        "WhatsApp (widget JoinChat)",
                        "WHATSAPP",
                    )

        for attr_name in ("data-whatsapp", "data-phone", "data-telephone"):
            raw = attr_map.get(attr_name)
            if raw:
                kind = "WHATSAPP" if attr_name == "data-whatsapp" else "PHONE"
                label_prefix = "WhatsApp" if kind == "WHATSAPP" else "Teléfono"
                self._add_signal(raw, label or f"{label_prefix} ({attr_name})", kind)
        # Treat block-level boundaries as whitespace so words don't run together.
        if tag in ("br", "p", "div", "li", "tr", "td", "h1", "h2", "h3", "h4"):
            self._chunks.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data)

    def text(self, max_chars: int) -> str:
        return " ".join(" ".join(self._chunks).split())[:max_chars]

    def contact_signals(self, source_url: str) -> list[dict[str, str]]:
        """Return explicit HTML contact values plus labelled visible phones.

        This deliberately runs on the complete parsed visible text, before the
        caller applies its model-token character cap. A bare digit string is not
        enough: visible numbers require a nearby contact label, while WhatsApp
        and ``tel:`` URLs are explicit contact mechanisms by themselves.
        """
        full_text = " ".join(" ".join(self._chunks).split())
        for match in _PHONE_TOKEN_RE.finditer(full_text):
            prefix = full_text[max(0, match.start() - 80) : match.start()]
            label_match = _PHONE_LABEL_RE.search(prefix)
            if not label_match:
                continue
            label = label_match.group(1)
            evidence = full_text[max(0, label_match.start()) : min(len(full_text), match.end() + 80)]
            self._add_signal(match.group(0), f"Teléfono ({label}): {evidence}", "PHONE")

        out: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for signal in self._contact_signals:
            key = (signal["kind"], _digits_only(signal["value"]))
            if not key[1] or key in seen:
                continue
            seen.add(key)
            out.append({**signal, "source_url": source_url})
        return out

    def _extract_href_contact(self, href: str, label: str) -> None:
        try:
            parsed = urlsplit(href)
        except ValueError:
            return
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        query = parse_qs(parsed.query)

        if scheme == "tel":
            self._add_signal(parsed.path, label or "Teléfono (enlace tel:)", "PHONE")
            return
        if scheme == "whatsapp":
            for value in query.get("phone", []):
                self._add_signal(value, label or "WhatsApp (enlace whatsapp://)", "WHATSAPP")
            return
        if host not in _WHATSAPP_HOSTS:
            return
        if host == "wa.me":
            self._add_signal(
                parsed.path.strip("/").split("/", 1)[0],
                label or "WhatsApp (enlace wa.me)",
                "WHATSAPP",
            )
            return
        for value in query.get("phone", []):
            self._add_signal(value, label or "WhatsApp (enlace de WhatsApp)", "WHATSAPP")

    def _add_signal(self, value: str, evidence: str, kind: str) -> None:
        value = value.strip()
        # A URL/query value can contain punctuation but must still contain a
        # phone-shaped sequence. The backend is authoritative for E.164.
        if len(_digits_only(value)) < 7:
            return
        self._contact_signals.append(
            {"value": value[:60], "evidence": evidence[:400], "kind": kind}
        )


def _digits_only(value: str) -> str:
    return "".join(char for char in value if char.isdigit())


def extract_text_links_and_contacts(
    html: str, max_chars: int, source_url: str
) -> tuple[str, list[str], list[dict[str, str]]]:
    """Return capped text, hrefs and pre-cap deterministic contact signals."""
    parser = _TextAndLinkExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 — malformed HTML must not kill a run
        logger.info("web_fetch_parse_partial", error=str(exc))
    return parser.text(max_chars), parser.links, parser.contact_signals(source_url)


def extract_text_and_links(html: str, max_chars: int) -> tuple[str, list[str]]:
    """Compatibility wrapper for callers that do not need contact signals."""
    text, links, _signals = extract_text_links_and_contacts(html, max_chars, "")
    return text, links


# ── URL / address validation ─────────────────────────────────────────────────


def _blocked_network_for(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> str | None:
    """Return the matching block-list CIDR, or None when the address is allowed."""
    # ``IPv6Address("::ffff:127.0.0.1").is_loopback`` is False, so unwrap first.
    if isinstance(address, ipaddress.IPv6Address):
        for embedded in (address.ipv4_mapped, address.sixtofour, address.teredo):
            if embedded is None:
                continue
            candidate = embedded[1] if isinstance(embedded, tuple) else embedded
            if isinstance(candidate, ipaddress.IPv4Address):
                hit = _blocked_network_for(candidate)
                if hit:
                    return hit

    for network in _BLOCKED_NETWORKS:
        if address.version == network.version and address in network:
            return str(network)
    return None


def parse_and_validate_url(raw: str) -> httpx.URL:
    """Parse ``raw`` and enforce the scheme/host policy (no DNS yet).

    Parsing goes through ``httpx.URL`` exactly once and every later decision uses
    ``url.raw_host`` (the IDNA-encoded form). Mixing parsers is a real
    vulnerability class here: ``httpx.URL.host`` *decodes* punycode back to
    Unicode, while ``socket.getaddrinfo`` applies permissive IDNA2003 which maps
    U+3002 (ideographic full stop) onto ".", so ``http://127。0。0。1/`` resolves
    to loopback. Using ``raw_host`` everywhere removes the differential.
    ``.raw_host`` also strips ``user@`` userinfo, which defeats
    ``https://expected.com@127.0.0.1/``.
    """
    try:
        url = httpx.URL(raw.strip())
    except Exception as exc:  # noqa: BLE001
        raise FetchBlocked("malformed_url", str(exc)) from exc

    if url.scheme not in ALLOWED_SCHEMES:
        raise FetchBlocked("bad_scheme", url.scheme or "(none)")

    host = (url.raw_host or b"").decode("ascii", errors="replace")
    if not host:
        raise FetchBlocked("no_host", raw[:120])
    if host.rstrip(".").lower() in _BLOCKED_HOSTNAMES:
        raise FetchBlocked("blocked_hostname", host)

    # Defense in depth: reject a blocked IP *literal* here as well, so a URL like
    # "http://127.0.0.1/" or a Location header pointing at one is refused at
    # parse time rather than relying solely on the resolve step. Short and
    # decimal forms ("127.1", "2130706433") are NOT valid for ip_address() and
    # fall through to _resolve_and_pin, which normalizes them via getaddrinfo and
    # validates the result — that remains the authoritative check.
    literal = host.strip("[]").split("%", 1)[0]
    try:
        address = ipaddress.ip_address(literal)
    except ValueError:
        pass  # a hostname, not a literal — validated after resolution
    else:
        blocked = _blocked_network_for(address)
        if blocked:
            raise FetchBlocked("blocked_ip", f"{literal} in {blocked}")

    return url


async def _resolve_and_pin(url: httpx.URL) -> str:
    """Resolve the host and return ONE validated IP to connect to.

    Every address the resolver returns is checked, and a single blocked entry
    rejects the whole host. That matters for round-robin DNS: validating only the
    address we happen to pick would let an attacker win by retrying.
    """
    host = (url.raw_host or b"").decode("ascii", errors="replace")
    port = url.port or (443 if url.scheme == "https" else 80)

    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchBlocked("dns_failure", f"{host}: {exc}") from exc

    if not infos:
        raise FetchBlocked("dns_failure", host)

    validated: list[str] = []
    for info in infos:
        sockaddr = info[4]
        raw_ip = str(sockaddr[0])
        # IPv6 sockaddrs can carry a %scope suffix that ip_address() rejects.
        raw_ip = raw_ip.split("%", 1)[0]
        try:
            address = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise FetchBlocked("unparseable_address", raw_ip) from exc

        blocked = _blocked_network_for(address)
        if blocked:
            raise FetchBlocked("blocked_ip", f"{host} -> {raw_ip} in {blocked}")
        validated.append(raw_ip)

    return validated[0]


class _PinnedTransport(httpx.AsyncHTTPTransport):
    """Send the request to a pre-validated IP while keeping TLS identity intact.

    The request URL's host is swapped for the literal IP so no second resolution
    can happen, but ``Host`` and ``sni_hostname`` still carry the real name — so
    certificate hostname verification continues to validate against the
    hostname, not the IP. Without ``sni_hostname`` we would have to disable
    verification and lose far more than we gained.
    """

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        pinned = request.extensions.get("pinned_ip")
        if pinned:
            host = (request.url.raw_host or b"").decode("ascii", errors="replace")
            port = request.url.port
            request.url = request.url.copy_with(host=pinned)
            request.headers["Host"] = (
                host if port is None else f"{host}:{port}"
            )
            request.extensions = {
                **request.extensions,
                "sni_hostname": host,
            }
        return await super().handle_async_request(request)


def build_client(timeout_seconds: float) -> httpx.AsyncClient:
    """An httpx client configured for pinned, proxy-free, single-use fetches.

    Two settings are load-bearing:

    ``trust_env=False``
        httpx otherwise honours ``HTTP_PROXY``/``HTTPS_PROXY``/``ALL_PROXY`` from
        the environment. If any is set in the task definition, the connection
        goes to the proxy, which re-resolves the ``Host`` header — silently
        voiding the IP pinning.

    ``max_keepalive_connections=0``
        httpcore keys pooled connections on ``(scheme, host, port)`` built from
        the *rewritten* (IP) URL. Two different hostnames pinned to the same IP
        therefore share one pool entry, so a request for ``b.com`` could be
        served over a connection whose certificate was validated for ``a.com``.
        Disabling keep-alive removes the collision.
    """
    return httpx.AsyncClient(
        timeout=timeout_seconds,
        follow_redirects=False,  # hops are validated manually
        trust_env=False,
        transport=_PinnedTransport(),
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=4),
    )


# ── Fetching ─────────────────────────────────────────────────────────────────

_REDIRECT_CODES = (301, 302, 303, 307, 308)


async def _read_capped(
    response: httpx.Response, max_bytes: int
) -> tuple[bytearray, bool]:
    """Read at most ``max_bytes`` (+ one chunk of overshoot) from ``response``.

    Streams via ``aiter_raw`` — deliberately NOT ``aiter_bytes``, which yields
    *decompressed* bytes: a ~1 KB gzip bomb can expand to tens of megabytes
    inside a single chunk, so the cap check would only fire after the allocation.
    We also send ``Accept-Encoding: identity``, so raw and decoded are the same
    thing on the happy path.

    Some responses arrive with the body already in memory (notably
    ``httpx.MockTransport`` in tests, which marks ``content=``-constructed
    responses as stream-consumed). In that case there is nothing left to stream,
    so slice what is already there — the cap is still honoured for what we
    return.
    """
    buffer = bytearray()
    if response.is_stream_consumed:
        body = response.content
        buffer += body[: max_bytes + 1]
        return buffer, len(body) > max_bytes

    async for chunk in response.aiter_raw(8192):
        buffer += chunk
        if len(buffer) > max_bytes:
            # Leaving the caller's context manager closes the connection.
            return buffer, True
    return buffer, False


async def fetch_page(
    client: httpx.AsyncClient,
    raw_url: str,
    *,
    max_bytes: int,
    max_chars: int,
    max_redirects: int = 3,
) -> FetchedPage:
    """Fetch one URL, following (and re-validating) up to ``max_redirects`` hops.

    Raises :class:`FetchBlocked` when policy refuses the URL or any hop.
    """
    seen: set[str] = set()
    url = parse_and_validate_url(raw_url)

    for _hop in range(max_redirects + 1):
        key = str(url)
        if key in seen:
            raise FetchBlocked("redirect_loop", key[:200])
        seen.add(key)

        pinned_ip = await _resolve_and_pin(url)

        async with client.stream(
            "GET",
            url,
            headers={
                "User-Agent": USER_AGENT,
                # Disable compression: aiter_bytes() would yield DECOMPRESSED
                # bytes, so a ~1 KB gzip bomb can expand to tens of MB inside a
                # single chunk and blow the cap before the check runs.
                "Accept-Encoding": "identity",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
            },
            extensions={"pinned_ip": pinned_ip},
        ) as response:
            if response.status_code in _REDIRECT_CODES:
                location = response.headers.get("location")
                if not location:
                    raise FetchBlocked("redirect_without_location", key[:200])
                # Resolve relative Locations against the current URL, then run
                # the FULL validation again on the next loop iteration.
                url = parse_and_validate_url(str(url.join(location)))
                continue

            if response.status_code >= 400:
                raise FetchBlocked("http_error", str(response.status_code))

            content_type = (
                response.headers.get("content-type", "").split(";")[0].strip().lower()
            )
            if content_type not in ALLOWED_CONTENT_TYPES:
                raise FetchBlocked("bad_content_type", content_type or "(none)")

            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise FetchBlocked("oversize", declared)

            buffer, truncated = await _read_capped(response, max_bytes)
            if truncated:
                logger.info("web_fetch_truncated", url=key, cap=max_bytes)

            html = bytes(buffer[:max_bytes]).decode(
                response.encoding or "utf-8", errors="replace"
            )
            text, hrefs, contact_signals = extract_text_links_and_contacts(
                html, max_chars, key
            )
            absolute = _absolutize(url, hrefs)
            return FetchedPage(
                url=key,
                text=text,
                links=absolute,
                contact_signals=contact_signals,
                bytes_read=len(buffer),
            )

    raise FetchBlocked("too_many_hops", str(url)[:200])


def _absolutize(base: httpx.URL, hrefs: Iterable[str]) -> list[str]:
    """Resolve hrefs against the FINAL (post-redirect) URL, dropping non-web ones."""
    out: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        candidate = (href or "").strip()
        if not candidate or candidate.startswith("#"):
            continue
        lowered = candidate.lower()
        if lowered.startswith(("mailto:", "tel:", "javascript:", "data:")):
            continue
        try:
            resolved = str(base.join(candidate))
        except Exception:  # noqa: BLE001 — a junk href is not an error
            continue
        if not resolved.lower().startswith(("http://", "https://")):
            continue
        resolved = resolved.split("#", 1)[0]
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


@dataclass
class SiteFetchResult:
    pages: list[FetchedPage] = field(default_factory=list)
    """Per-reason counts of refused URLs, for metrics + the attempt's metrics blob."""
    blocked: dict[str, int] = field(default_factory=dict)
    timed_out: bool = False

    @property
    def combined_text(self) -> str:
        return "\n\n".join(f"[{p.url}]\n{p.text}" for p in self.pages)

    @property
    def urls(self) -> list[str]:
        return [p.url for p in self.pages]


# Link text / path fragments worth crawling, best first.
#
# The head of the list targets a PHONE NUMBER; the tail targets QUALIFICATION —
# how big the business is and what it sells. Order matters: the crawl stops at
# `sherlock_max_pages`, so contact pages must stay ahead of catalogue pages, and
# a prospect with no phone anywhere is still worth sizing.
_PRIORITY_PATH_HINTS = (
    "contacto",
    "contactenos",
    "contactanos",
    "contact",
    "quienes-somos",
    "nosotros",
    "about",
    "acerca",
    "empresa",
    "sedes",
    "ubicacion",
    # Size and offering signals: branch lists, service catalogues and price
    # pages are where "3 sedes" and "pedidos por WhatsApp" actually appear.
    "sucursales",
    "servicios",
    "productos",
    "planes",
    "precios",
    "tarifas",
    "tienda",
    "menu",
)


def prioritize_links(base: httpx.URL | str, links: Iterable[str]) -> list[str]:
    """Same-site links most likely to carry contact details, best first."""
    scored: list[tuple[int, str]] = []
    for link in links:
        if not same_site(base, link):
            continue
        lowered = link.lower()
        for rank, hint in enumerate(_PRIORITY_PATH_HINTS):
            if hint in lowered:
                scored.append((rank, link))
                break
    scored.sort(key=lambda pair: pair[0])
    seen: set[str] = set()
    out: list[str] = []
    for _rank, link in scored:
        if link not in seen:
            seen.add(link)
            out.append(link)
    return out


async def fetch_site(
    start_url: str,
    *,
    max_pages: int,
    max_bytes: int,
    max_chars: int,
    per_request_timeout: float,
    total_budget_seconds: float,
    extra_paths: Iterable[str] = (),
) -> SiteFetchResult:
    """Fetch a start URL plus a bounded set of same-site contact-ish pages.

    ``total_budget_seconds`` is enforced with ``asyncio.timeout`` around the whole
    operation, because httpx timeouts are strictly per-operation: a server
    trickling one byte every ``read``-timeout-minus-epsilon seconds would
    otherwise stream for an unbounded total time. Whatever pages completed before
    the deadline are returned with ``timed_out=True`` rather than being discarded.

    ``extra_paths`` lets the caller's refinement loop suggest additional
    candidates (absolute URLs or paths); they are still subject to the same-site
    rule and to full per-URL validation.
    """
    result = SiteFetchResult()

    def note_blocked(reason: str) -> None:
        result.blocked[reason] = result.blocked.get(reason, 0) + 1

    async def run() -> None:
        async with build_client(per_request_timeout) as client:
            try:
                first = await fetch_page(
                    client, start_url, max_bytes=max_bytes, max_chars=max_chars
                )
            except FetchBlocked as exc:
                note_blocked(exc.reason)
                logger.info(
                    "web_fetch_blocked",
                    url=start_url,
                    reason=exc.reason,
                    detail=exc.detail[:200],
                )
                return
            result.pages.append(first)

            # Candidate order: caller hints first (the refinement loop has already
            # reasoned about them), then links discovered on the landing page.
            candidates: list[str] = []
            for path in extra_paths:
                try:
                    candidates.append(str(httpx.URL(first.url).join(path)))
                except Exception:  # noqa: BLE001
                    continue
            candidates.extend(prioritize_links(first.url, first.links))

            visited = {first.url}
            for candidate in candidates:
                if len(result.pages) >= max_pages:
                    break
                if candidate in visited or not same_site(first.url, candidate):
                    continue
                visited.add(candidate)
                try:
                    page = await fetch_page(
                        client, candidate, max_bytes=max_bytes, max_chars=max_chars
                    )
                except FetchBlocked as exc:
                    note_blocked(exc.reason)
                    logger.info(
                        "web_fetch_blocked",
                        url=candidate,
                        reason=exc.reason,
                        detail=exc.detail[:200],
                    )
                    continue
                result.pages.append(page)

    try:
        async with asyncio.timeout(total_budget_seconds):
            await run()
    except (asyncio.TimeoutError, TimeoutError):
        result.timed_out = True
        logger.info(
            "web_fetch_budget_exhausted",
            url=start_url,
            budget=total_budget_seconds,
            pages=len(result.pages),
        )
    except Exception as exc:  # noqa: BLE001 — a fetch failure is a NO_RESULT, not a crash
        logger.warning("web_fetch_failed", url=start_url, error=str(exc))

    return result


def site_key(raw: str) -> str:
    """Registrable domain of ``raw``, counting PRIVATE public suffixes.

    A dedicated extractor rather than ``agents.dedup.normalize_domain``, because
    the two need different rules and dedup's are load-bearing elsewhere:

    * dedup uses ICANN suffixes only, so ``user1.github.io`` collapses to
      ``github.io``. That is fine for "have we already got this company in the
      CRM?", and changing it would silently alter prospecting's dedupe results.
    * a crawl boundary must NOT collapse it. Shared-hosting platforms
      (``github.io``, ``wixsite.com``, ``myshopify.com``, …) put unrelated
      businesses on sibling subdomains, so treating them as one site would let a
      discovered link pull in a different company's pages and attribute them to
      this contact.

    ``include_psl_private_domains=True`` uses the PSL's private section, which is
    exactly the list of "these subdomains are separate owners".
    ``suffix_list_urls=()`` pins it to the bundled snapshot — no network at
    runtime, deterministic in tests (same rationale as ``dedup._EXTRACT``).
    """
    ext = _SITE_EXTRACT((raw or "").strip().lower())
    if not ext.domain:
        return ""
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


# Kept as the private name the crawl code has always used.
_site_key = site_key


def site_label(raw: str) -> str:
    """The registrable domain's own label, without its public suffix.

    ``https://colegiosanjose.edu.co/`` -> ``colegiosanjose``; on shared hosting
    ``https://colegio.wixsite.com/x`` -> ``colegio``, because the PRIVATE suffix
    list makes ``wixsite.com`` the suffix. That is exactly the part a business
    puts its own name in, so it is what website discovery matches a contact name
    against. Empty when no domain can be determined.
    """
    return _SITE_EXTRACT((raw or "").strip().lower()).domain or ""


def same_site(a: httpx.URL | str, b: httpx.URL | str) -> bool:
    """True when two URLs belong to the same site owner.

    Registrable domain, not exact host: ``acme.com`` linking to ``www.acme.com``
    or ``contacto.acme.com`` is the overwhelmingly common real-world shape for
    contact pages, and exact-host matching would discard exactly the pages we
    came for.

    This is a RELEVANCE boundary, never a security boundary —
    ``internal.acme.com`` can point at 10.0.0.5. Every discovered link goes
    through the full scheme/DNS/IP validation independently, so a same-site link
    is never trusted on the strength of this check alone. Fails CLOSED when
    either domain cannot be determined.
    """
    left = _site_key(str(a))
    right = _site_key(str(b))
    if not left or not right:
        return False
    return left == right
