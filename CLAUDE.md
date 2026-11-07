# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the service (dev, with hot reload)
uv run uvicorn src.main:app --reload --port 8000

# Run tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_graph.py -v

# Run a single test class or method
uv run pytest tests/test_graph.py::TestGraphWiring::test_graph_routes_faq_to_faq_response -v

# Run with Docker Compose
docker compose up agent --build

# LangGraph Studio (visual graph debugging, in-memory checkpointer)
docker compose up langgraph-studio --build
# Opens at http://localhost:2024
```

## Architecture

Helena Agent is a **LangGraph-based multi-agent service** for WhatsApp customer attention. A NestJS backend sends chat messages via HTTP; this service processes them through a state machine and streams responses as Server-Sent Events (SSE). State is persisted per conversation in PostgreSQL using LangGraph's `AsyncPostgresSaver`. All LLM calls use the OpenAI API (model: `gpt-5`) via `AsyncOpenAI` — the API key is passed per-request from NestJS or falls back to `OPENAI_API_KEY` env var. All user-facing responses are in Spanish.

### Request Flow

```
NestJS → POST /chat/stream (service credential + message body)
  → FastAPI (src/main.py): authenticate the calling service, scope thread_id as
    "{tenantId}:system:{conversationId}:{agentCodeName}:v{agentVersion}"
  → per-thread coalescing (_coalesced_stream): serialize runs, merge rapid message bursts
  → LangGraph graph.astream() → PostgreSQL checkpointer
  → SSE events streamed back: token | step_progress | execute_workflow | node_update | interrupt_detected | done | error
```

### Multi-message handling

WhatsApp users often split one thought across several rapid messages; NestJS forwards each as its own `POST /chat/stream`. Two layers turn a burst into ONE coherent turn:

1. **Per-thread coalescing** (`_coalesced_stream` in `src/main.py`): runs are serialized per `thread_id` with an in-process `asyncio.Lock`. A turn waits until no new fragment has arrived for `MESSAGE_SETTLE_SECONDS` (default 2s, total wait capped at `MESSAGE_SETTLE_MAX_SECONDS`, default 10s), then drains all buffered fragments into a single `HumanMessage` (texts joined with newlines, attachments concatenated, per-request FAQ payloads unioned with question-level dedupe). Superseded requests complete with an empty `done` event (`turn_usage: []`, `coalesced: true`) — NestJS treats that as "no reply": it releases the credit reservation and sends nothing. Locks are process-local; horizontal scaling requires sticky routing by thread_id. NestJS complements this with burst-aware FAQ retrieval: `AgentService.buildFaqQueryText` queries FAQs with the whole trailing inbound burst, not just the current fragment.
2. **Trailing-burst readers** (`latest_user_messages` / `latest_user_text` in `src/agents/utils.py`): every node that consumes "the user's message" reads ALL trailing consecutive human messages since the bot's last reply — never `messages[-1]` alone — so fragments that land as separate checkpoint entries are still answered together. Keyword confirmation checks (booking_confirm, school_graph question guard) test each fragment individually.

### Graph State Machine (`src/graphs/main_graph.py`)

```
START → triage
  → sales_collect    → order_summary → execute → END
  → tracking_collect → execute → END
  → complaint_collect → execute → END
  → faq_response     → END
```

- **triage** (`src/agents/triage.py`): Silent LLM intent classification — `sales | tracking | complaint | faq`. Skips re-classification if a flow is already in progress. Also contains `route_from_triage()` which resumes the flow at the correct phase based on state flags.
- **sales_collect** (`src/agents/sales_collect.py`): 3 conversational steps collecting order data via two LLM calls per turn (extraction → JSON, then conversational reply). Emits `step_progress` SSE. Steps: (1) customer info, (2) products/quantities from catalog, (3) delivery details.
- **order_summary** (`src/agents/order_summary.py`): Presents order summary and waits for keyword confirmation (`confirmar`, `sí`, `ok`, `dale`, etc.).
- **tracking_collect** (`src/agents/tracking_collect.py`): Collects order ID or customer details for status lookup. Single step.
- **complaint_collect** (`src/agents/complaint_collect.py`): Collects complaint details: order ref, issue description, desired resolution. Single step.
- **faq_response** (`src/agents/faq_response.py`): Answers FAQs (hours, location, payments, shipping) and serves as fallback for unknown intents.
- **query_normalizer** (`src/agents/query_normalizer.py`): First node in every graph (`START → query_normalizer → triage`). Usually a pure-Python passthrough that stamps `original_text`; the LLM typo-correction runs only when the deterministic trigger fires (`should_normalize` in `src/graphs/shared_routing.py`: per-tenant flag `query_normalization_enabled` from the request AND env kill switch `QUERY_NORMALIZATION_ENABLED` on, text ≥ 4 chars, no `ESCALATION_KEYWORDS` hit, initial backend FAQ retrieval empty or max `score` < `FAQ_SCORE_TRIGGER_THRESHOLD`=0.15). On an applied correction (risk LOW/MEDIUM + confidence ≥ 0.7) it re-fetches FAQs once via `backend_client.search_faqs` (`GET /internal/faqs/search`, conversation-scoped) and overwrites `state["faqs"]`; it NEVER rewrites `messages` — the corrected text lives only in the write-once `normalization` provenance dict (`{enabled, model, confidence, changed_meaning_risk, reason, applied, corrected_text}`, sole writer: this node). Camila guardrails: the deterministic identity short-circuit requires the conflict to persist on both original and corrected transcripts when a normalization applied (`_identity_conflict_confirmed`), and a weak LLM `identity_conflict` (confidence < 0.75) on a normalized turn without escalation keywords is downgraded to `faq`. Observability: `query_normalization` structured log line + `query_normalizations_total{outcome}` counter.
- **greeting_response** (`src/agents/greeting_response.py`): Greets known contacts (triage intent `greeting` + contact name on file). LLM-rendered from the tenant's `{AGENT_TYPE}_GREETING` prompt (resolved via `resolve_prompt`, placeholders `{persona}`/`{role}`/`{name}`/`{language_rule}` substituted with a brace-tolerant `format_map`); falls back to the deterministic per-language templates when the LLM call fails or returns empty (fallback contributes no `turn_usage`). Shared by every agent graph.
- **execute** (`src/agents/execute.py`): Emits `execute_workflow` SSE for NestJS to trigger backend workflows. Sets `execute_confirmed=True` to prevent re-execution.

**Auto-chaining**: When a node completes all its sub-steps, the next node runs in the same turn — no extra user message needed. This is implemented via conditional edges (`_route_from_sales_collect`, `_route_from_order_summary`, `_route_from_tracking_collect`, `_route_from_complaint_collect`) that check state flags.

### Email agent (`clara`, agent type EMAIL)

Gmail → sanitize → triage → task extraction → reply draft → **human approval in the CRM** → Gmail **draft**. It never sends. Ported from the `emailAi` prototype.

- **Trigger:** the backend's email sweep (Cloud Scheduler) calls `POST /email/sync` (202, background) per mailbox and `POST /email/follow-up` per due thread; `POST /email/drafts` runs synchronously after a human approved/edited a reply. `agent_code_name` is **required** on all three (no default) and must resolve to a graph compiled with `name=EMAIL_GRAPH_NAME`, else 400.
- **Graph** (`src/graphs/clara_graph.py`, nodes in `src/agents/clara/nodes.py`): `precheck → triage → (extract_task) → draft_reply`, or `draft_follow_up` in follow-up mode. **Tool-free**: each node is pure Python or ONE `generate_structured` call. `precheck` short-circuits only on strong signals (`Auto-Submitted`, bounces, internal sender = same non-public domain as the mailbox) and fails open otherwise. Guards run after every node: parser `security_flags` force human review; `evidence_quote`/`due_date_evidence` must be literal substrings of the email.
- **Side effects** live only in `src/agents/clara/runner.py`: access token from the backend (scopes must be exactly `gmail.readonly` + `gmail.compose`, checked on every use — `src/services/gmail.py`), history-cursor sync, reports to `/api/v1/internal/email/*`. Idempotency: `alreadyIngested` before any model call, one checkpoint thread per message (`{code}:{tenant}:{mailbox}:{key}`) resumed by `run_graph_once` and **deleted** after the backend confirms, `X-Clara-Approval-Id` header so a retried draft is found instead of duplicated. A message that keeps failing is reported FAILED until the backend answers `giveUp` (3 attempts); a provider config error fails the whole run. Credentials never fall back to the platform OpenAI key.
- **Parser** (`src/services/email_parser.py`): hidden HTML, quoted history, zero-width/bidi characters (stripped before the injection regexes) and injection phrases are removed or flagged before anything reaches a model. Email text reaches prompts only inside `<<<CORREO … CORREO>>>` fences.
- **Prompts** (`src/agents/clara/prompts.py`, keys `EMAIL_*`) are mirrored in the backend's `DEFAULT_PROMPTS`; every usage row carries `prompt_key/prompt_id/prompt_version/prompt_sha` and `latency_ms` (optional `InvocationUsage` keys only this agent sets).
- **Structured output:** `ChatProvider.generate_structured` (prompt-JSON fallback); Gemini overrides it with native `response_schema` + per-node `ThinkingConfig`, folds thinking tokens into `output_tokens` (the backend bills reasoning as a subset of output) and retries once without a budget on models that cannot disable thinking.
- **Memory:** human-edited replies (last 3 per sender, keyed by approval id) in the existing `AsyncPostgresStore`, namespace `("clara_sender_prefs", tenant_id)`. Written only on a human edit, never by a model.
- **Eval:** `uv run python scripts/eval_clara.py` (real LLM, costs money) over `tests/fixtures/email/` with a local LLM-as-judge for `draft_criteria`.

### Prospecting search sources (aurora)

- **Sources:** organic results from Serper `/search`; business listings from **Google Places API (New)** text search (`src/services/google_places.py`, `GOOGLE_PLACES_API_KEY`), which replaced Serper `/places` on 2026-10-16 and halves the Serper credits a run spends. `PROSPECTING_PLACES_PROVIDER=google|serper|off`; "google" without a key falls back to Serper. Places rows carry `locality` from `addressComponents`, which `extract_and_enrich` prefers over the LLM's city.
- **An empty account is a failure, not an empty result.** Serper answers `400 {"message":"Not enough credits"}`; `serper_post` raises `SerperQuotaExhaustedError` (a `SerperAuthError` subclass) on it without retrying, and any other 4xx is a per-query miss that is never retried (Serper bills retries). The run wrapper reports `FAILED` with `metrics.reason` `search_quota_exhausted` / `search_auth_failed` plus `provider` (`main.prospecting_failure`) — a contract with the backend, which opens one platform incident and pauses every tenant's prospecting. Before this (2026-10-13 → 10-16) every run reported COMPLETED with nothing found.
- **`GET /prospecting/search-health`** (service auth, bills nothing): `{serper: {ok, balance}, google_places: {ok, configured}}`. The backend reads it before its daily sweep while a search incident is open and resolves the incident once the provider reports `ok: true`.
- **Credit control:** `PROSPECTING_MAX_SERPER_CALLS_PER_RUN` (60) stops issuing queries; `report_node` reads the free `GET /account` balance into `metrics.serper_balance` and warns `serper_low_balance` under `PROSPECTING_SERPER_LOW_BALANCE`.
- **What never reaches extraction:** directories, job boards, document dumps, maps and `.gov.co` (`_BLOCKED_DOMAINS`, extended by `PROSPECTING_BLOCKED_DOMAINS`) — a listing keeps its row but loses the directory URL. Social pages stay but skip the doomed fetch (`_NO_FETCH_DOMAINS`).
- **Region:** in Colombia a configured core city also covers its metropolitan municipalities (`METRO_AREAS`: Envigado → Medellín, Soacha → Bogotá, Yumbo → Cali…).
- Run metrics added: `places_calls`, `places_provider`, `blocked_results`, `extract_failed`, `extract_rate_limited`, `serper_balance`. Tests: `tests/test_prospecting_search_sources.py`.

### Theology agent (`ismael`, agent type THEOLOGY)

Christian theology Q&A for the MOODLE bubble, answered by **Company Brain** (the cited RAG in `yorch-tauri-backend`, one EC2 host in AWS that is **stopped when idle**).

- **Graph** (`src/graphs/ismael_graph.py`, nodes in `src/agents/ismael/nodes.py`): `ismael_triage → {ismael_survey | ismael_pending | ismael_start | ismael_moodle_support | ismael_teacher | ismael_faq | greeting_response | ismael_closing | ismael_rating | ismael_off_topic} → END` (`ismael_teacher` may hand over to `ismael_start`, `ismael_moodle_support` or back to `ismael_triage`). Triage is deterministic while a survey question is open or a job is running; otherwise ONE `generate_structured` call returns the intent and the question rewritten to stand alone (Brain never sees the conversation). No name capture — MOODLE identity is server-attested.
- **The answer never comes from the graph.** A cold Brain plus a real question outlasts the backend's 60 s turn, so `ismael_start` replies with a holding message and `rag_job.spawn` runs a background task: `brain.ensure_started` → `wait_healthy` (re-requests the start every 30 s: a host caught mid-stop refuses StartInstances) → `POST /ask` → poll `GET /ask/:id` → `POST /api/v1/internal/conversations/:id/agent-messages` (idempotent on `jobId`; the backend stores, dispatches over the widget SSE and bills `turnUsage`). `answered` is sent **verbatim** plus a References block from `citations`+`evidence` (an LLM rewrite could detach a claim from its source); `insufficient_evidence`/`off_corpus` get a ≤120-word general answer (`THEOLOGY_GENERAL`), sent without any notice (the job log's `general:<state>` outcome still tells it apart); boot/answer timeout or a failed question gets an apology.
- **Survey (statistics only, once per contact):** on a new contact's first question `ismael_start` fires the EC2 start in the background and asks 3 fixed questions (`texts.SURVEY`: level, topic, intendedUse — option keys are statistics, never tenant-editable); the answers go to `POST /internal/contacts/:id/ismael-survey` → `profileData.ismael`, and the backend then reports `user_context.ismael_survey_done`. Numbers/labels parse deterministically; an LLM is asked only otherwise; unreadable → `no_answer`, never re-asked; a new question typed instead of an answer replaces the pending one.
- **One job per conversation:** running state in the shared `AsyncPostgresStore` (`("ismael_jobs", tenant)`, key conversationId, TTL = boot + answer timeout + 2 min) so it holds across Cloud Run instances; an in-process dict is the fallback. A recycled instance loses its job — the entry expires and the student can ask again.
- **Starting the host** (`src/services/brain.py`): metadata-server ID token (audience `BRAIN_OIDC_AUDIENCE`) → boto3 `AssumeRoleWithWebIdentity` → `ec2:StartInstances`. No AWS key anywhere. `BRAIN_START_MODE=skip` off Cloud Run. Brain auth is `X-Api-Key` (a Brain service key bound to one tenant).
- **Moodle support and contacting a teacher:** triage also returns `moodle_support` / `contact_teacher` (+ `teacher_topic`); these are answered even while a Brain job runs. `src/agents/ismael/support.py` answers platform questions with one `generate_structured` (`THEOLOGY_MOODLE_SUPPORT`) and appends the guided-ticket block in code from `user_context.lms_support_*`. `src/agents/ismael/teacher.py` is a latched state machine in `state["ismael"]["teacher"]` (30 min TTL): offer help first → courses/teachers from the backend → send or link → draft → explicit "Enviar" → backend → Moodle. Free-text replies are matched deterministically first (`common.match_option`), LLM otherwise; a different question drops the flow and re-triages the same turn (`ismael_route = "retriage"`). Shared helpers in `common.py`.
- **Ask before escalating** (`support.py`, after the 2026-10-12 review of Casa Roca's conversations): a support case lives in `state["ismael"]["support"]` (`question`, `awaiting`, `clarified`, 30 min TTL like every ismael latch, `common.latch_fresh`). Every escalation — `needs_ticket`, `redirect` and the new `teacher` outcome — is **held in code** (`hold_escalation`) until the student answered one clarifying question; the model's `clarifying_question` (or `texts.support_clarify`) is sent instead, also on a timeout. Steps that were given count as the back-and-forth, so «no me funcionó» may escalate. Triage sends an `other` reply back into an open case. A disputed grade or quiz question is outcome `teacher`: support hands over to `ismael_teacher` with `teacher.from_support(reason)`, which skips the "can I help first?" offer and prefixes `teacher_from_support` — never the technical form.
- **Closing and the service rating** (`closing.py`): a bare thank-you/acknowledgement is caught by `common.is_bare_closing` before the triage LLM (works with an old tenant prompt), and the `closing` intent covers the rest; a greeting after ismael already replied goes to the same node (`regreet`), never the introduction again. Once per 24 h and only after ismael helped (`mark_helped` sets `helped_at` on answers, FAQs, tickets, the teacher's contact), the closing ends with «👍 Me ayudó / 👎 No me ayudó» quick replies; the answer (latch `ismael["rating"]`, routed without triage) is saved via `backend_client.save_ismael_rating` → `POST /internal/conversations/:id/ismael-rating`, stored as `message_feedback` on the closing question. 👎 asks once for the reason and saves it on the same message (`messageId` from the first save). Another question instead of a rating is re-triaged in the same turn. Saving never blocks the reply.
- **Institution FAQs** (`src/agents/ismael/faq.py`): triage gets this turn's FAQ candidates appended to its prompt and returns `faq_id`; a validated match routes to `ismael_faq` (one `generate_structured`, `THEOLOGY_FAQ`, verbatim fallback, `faq_used` set). A covered theology question never reaches Brain. The chosen FAQ is stored whole in `state["ismael"]["faq"]` since `state["faqs"]` changes every turn (the teacher flow hands it over later).
- **The ticket once per case, and a failing form** (`support.py`, after the 2026-10-21 review): `case["ticket_given_at"]` — a later `needs_ticket` in the same case appends `texts.ticket_already_given` instead of the block. Outcome `form_failed` (the institution's support form itself errors) answers with fixed tips in code (`form_failed_message`: screenshot not photo, format/size from `lms_support_instructions`, digits-only ID, reload, other browser) and never re-pastes the ticket; `reports_form_failure` forces it even when the model says `needs_ticket` once a ticket was given (a bare «Sale error» stays in that state). `lms_teacher_messaging: "false"` (sent by the backend when the connection switched messaging off) adds «no sugieras la mensajería de Moodle» to the support data block.
- **Looking an error up on the web** (`src/agents/ismael/research.py`): when the first support call sets `research_query` (a concrete Moodle error it cannot explain) or the text is a pasted Moodle error page («Más información sobre este error»), and that call took ≤ 8 s: one Serper query on `docs.moodle.org` alone (an error code like `error_question_answers_missing_in_db` is searched by itself — docs has a page per code; OR-ing `site:`s returned forum index pages), then `moodle.org`/`tracker.moodle.org` only if docs found nothing; forum index pages and duplicate doc versions dropped; ≤ 2 pages. **Every moodle.org host answers our fetcher with a Cloudflare challenge (403)**, so a refused page is read through Serper's scrape endpoint (`SCRAPE_ENDPOINT`, ~10 credits) — only the top one (`ISMAEL_RESEARCH_SCRAPE_PAGES`=1), the rest keep their snippet. A second `generate_structured` (`ResearchedSupportResult`, fixed `RESEARCH_PROMPT`, pages fenced `<<<FUENTE … FUENTE>>>`) answers from them; its `source_url` must be a page actually read, the link is appended in code («Más información: …»), and a cited answer is never held back as a ticket. Otherwise the first answer stands. Once per case; search + read share `ISMAEL_RESEARCH_TIMEOUT_SECONDS` (12), the second call 15 s. Fail-open on no key/credits/timeout. **Cached platform-wide** in the shared `AsyncPostgresStore` (namespaces `("ismael_research", "searches"|"pages")`, keys = sha256 of the error code / folded query and of the URL; public docs, so one tenant's lookup serves all): a search and a page's text for `ISMAEL_RESEARCH_CACHE_DAYS` (30), a search that found nothing for `ISMAEL_RESEARCH_NEGATIVE_CACHE_DAYS` (7); a failed search or no credits is never cached, nor is a snippet as a page, nor the model's answer. Age is checked in code (`stored_at`); store down → an in-process dict (256 entries). A repeated error costs 0 credits (outcome `cached`). Log `ismael_research` (`cached`), counters `ismael_research_total{outcome}` and `ismael_research_cache_total{kind,result}`.
- **Teacher flow: the course they named** (`teacher.match_courses`): without a current course, a course whose subject (the `fullname` before the first «-», trailing level dropped) appears as whole words in the reason, the question or the message is picked without asking; several matches narrow the options. A course with no listed teacher gets tips (`teacher_none`, Participantes filtered by role), never a dead end.
- **Asking about one book** (`rag_job.match_work`): triage returns `work` (the book/author a theology question names). The job reads Brain's catalog (`brain.list_documents`, `GET /libraries/:id/documents`, cached 1 h — the service key reaches it since Brain `feat/ismael-catalog`) and, when one document matches (title contained either way, ≥ 2 words or 8 letters; an author alone only with a single book), asks with `filters: {document_id}`. Brain never searches titles or authors, so «¿y el libro Las llaves del poder?» was `insufficient_evidence` with the book in the library. A catalog error asks unscoped. `GENERAL_PROMPT` never describes a work it did not find. `ismael_brain_answer` logs `state`, Brain's `reason`, evidence titles and the filter. References show `clean_title` («07-LlavesDelPoder-INT.pdf.corrected» → «Llaves Del Poder»). **Brain's ficha (2026-10, `feat/ismael-ficha`)**: the catalog also carries `subtitle`, `contributors` (names), `content_type`, `published_year` — `match_work` matches title+subtitle and any contributor's surname. Each evidence item carries the document's `author`/`source_name`/`published_year`, so a reference reads «Las llaves del poder — Darío Silva-Silva (2004), p. 12»; a recording's detail is parsed from its timed locator («min 12:34 · <watch URL>», an `s3://` URL is left out, ≤ 200 chars); a ficha citation has no detail. An older Brain sends none of it and the references look as before; `clean_title` stays as the fallback.
- Tests: `tests/test_ismael_graph.py`, `tests/test_ismael_moodle.py`, `tests/test_ismael_research.py`.

### Assessment agent (`pablo`, agent type ASSESSMENT)

Conversational assessments inside the Moodle activity `mod_vervuxassess`. Plan: `PABLO_ASSESSMENT_PLAN.md`. Phase 1 (this agent), phase 2 (`yorchio-backend/src/assessment/`) and phase 3 (`yorchio-moodle/vervuxassess/`) are built; local end-to-end with the real agent and a fake LLM: `yorchio-moodle/README.md` → «Pruebas locales de mod_vervuxassess».

- **The backend drives every turn.** `ChatStreamRequest.assessment_turn` (`{kind: start|answer|continue|timeout, attempt_id, assessment_id, cycle_index, total_cycles, modality, is_last_cycle?, scope?, answer?: {text, audio_url, timed_out}}`) arrives in `domain_state`, which is reset every request; `pablo_triage` routes on `kind` alone, so a student's text never steers the flow. Persistent state is `state["pablo"]` (current cycle only; responses, scores and feedback live in the backend). One thread = one attempt; a turn for another attempt goes to `pablo_error`.
- **No coalescing** (`_UNCOALESCED_CODE_NAMES` in `main.py`): turns are events, so `_serialized_stream` runs each request on its own (still serialized per thread, no settle window).
- **The backend owns the clock.** Every pablo reply is sent with the timer paused; custom SSE events `{"type": "assessment", "event": …}` tell the backend what happened: `cycle_presented` (restart the timer), `response_recorded` (`is_last` → show the survey), `timed_out`, `compose_failed` / `answer_failed` (timer stays paused). A cycle is reported to the backend **before** it is shown; a cycle the backend did not store is never shown.
- **Composition** (`compose.py`): the backend sends the authorised candidates for the next outcome; one `generate_structured` (tenant model) picks 2–3 and rewrites them. Ids are validated in code, the candidates go verbatim if the model fails, the rubric is never shown to the composer, and `component_id` = `{snapshot}:{rubric component}` (a rubric shared by several questions repeats its component ids).
- **Audio** (`transcription.py`): Speech-to-Text v2 synchronous `recognize` over REST with ADC (≤ 60 s of audio), 3 attempts; a failure is reported `transcriptionStatus: failed`, never raised.
- **Scoring + feedback** (`POST /assessment/score`, 202, after the UX survey): one structured call per answered cycle; answers fenced `<<<RESPUESTA … RESPUESTA>>>` (the closing marker is stripped from the body); points clamped in code; unanswered cycles score 0 without a call; a failed transcription scores 0 with `needsReview`; a model failure after `PABLO_SCORE_RETRIES` reports `scoring_failed` (never a silent zero). Teacher calibration examples are matched by `(rubric_version_id, rubric_component_id)`, ≤ 5 each. `mode: replay` reports points only (for approving a prompt draft). Feedback percentages are computed in code. De-duplicated per process on `attempt:scoring_version:mode:replay_id`.
- **Class report + improvement proposals** (`POST /assessment/insights`, 202): the backend sends the numbers; the model writes the narrative and proposes. Kept in code only: rubric clarifications for components the teacher adjusted ≥ 3 times in the same direction, question flags for ids that were sent, prompt guidance that quotes no teacher rationale. Everything is a proposal the backend stores for a human to approve.
- **Every structured call appends `common.json_contract(Schema)`** ("Responde SOLO con un objeto JSON…" + the JSON schema) to the system prompt in code. Only Gemini enforces a schema natively; OpenAI/Anthropic use the prompt-JSON fallback, so without it every pablo call failed on those providers — and a tenant-edited prompt could drop the line. The prompt's provenance sha is computed before the contract is appended.
- **Conversational shell** (`feat/pablo-conversational`, plan in `~/.claude/plans/escribe-un-plan-hazme-twinkly-kurzweil.md`): turn kinds `open` (greeting by `student_name` — Moodle's first name, never derived from the full name — what the conversation is about, `topics`, the warm-up offer when `warmup_available`; event `opening_presented`), `warmup` (the teacher's `warmup_prompt` verbatim; `warmup_presented`) and `warmup_answer` (transcribed as the microphone test, acknowledged like a real answer; `warmup_recorded`). **The warm-up is not a cycle**: it never touches `cycle`/`current_cycle`/`used_question_ids` and reports nothing to the backend — it exists only in the conversation. The cycle header is narrative (`compose.progress_line`: "Seguimos con el segundo tema de tres: la gracia", "Volvemos al tema «la fe»…", "Es la última pregunta.") from the backend's `outcome_position`/`outcome_count`/`round`; an older backend keeps "Pregunta n de total". The acknowledgement (`echo.py`, prompt `ASSESSMENT_ACK`, background model, 6 s) quotes a fragment of the typed answer **only if code finds it literally in the answer**; the thanks is fixed and chosen by `cycle_index`, never by the answer; an empty answer gets neither. The closing before the survey is a neutral recap of `state["pablo"]["outcomes_seen"]`; "what you showed / what to explore" is the feedback's `overall_summary`, now spoken and in second person.
- **Follow-up question** (same branch, delivery B): when the backend sends `followup_allowed` on an `answer` turn (the activity has it on, the plugin is conversational, the attempt has time left) and the answer has content (typed or a transcript that worked), pablo asks **one** follow-up — to every such student, never depending on how good the answer was. One structured call (`followup.py`, prompt `ASSESSMENT_FOLLOWUP`, tenant model, 8 s) returns `{quote, target_component_id|null, mode: gap|deepen, question}` — the component the answer covers least, never points; it is the only chat-time call that sees the rubric (names + criteria kept in `cycle.rubric_criteria`, never calibration examples). Code checks the target, the length and the `?`, and the **leak guard** refuses a question with a content word (≥ 5 letters, folded) found in the rubric but neither in the question the student saw nor in their answer (`_GENERIC` words such as «ejemplo» excepted); a refused question (reason logged `pablo_followup_refused`: `too_long` >400, `no_question`, `leak:<words>`, `failed`) is asked for ONCE more with the reason (`refusal_note`); twice refused → a question in code on the student's own words (`followup_quoted_{0,1,2}` by cycle, quote = the model's validated quote or the answer's first clause), and the generic one (no «Parte A» with one question) only when there is nothing to quote; provenance `fallback: quoted|generic` + `refusal`. A tail after the last «?» is dropped. `MOVES_RULE` (appended in code) varies the question by the answer — the reason behind a claim, a testing case, linking two ideas, pinning a term; an example is one option. HTML entities are decoded on input and on everything shown (`common.plain`). Reported before it is shown (`POST …/cycles/:n/followup`), then event `followup_presented`; turn kind `followup_answer` (`pablo_followup_answer`) reports `…/followup/response` and ends the cycle (`response_recorded`). If the backend did not store it, the cycle ends as before. Scoring: `ScoreCycle.followup` — both answers fenced, scored together, the follow-up worth the same (`SCORE_PROMPT`); `common.answer_text` covers both.
- **Information for graders** (`feat/pablo-grader-info`, plan `PABLO_GRADER_INFO_PLAN.md`): the essay's `graderinfo` arrives as `grader_info` on each scoring question and each next-cycle candidate. It is **never** shown to the composer (`candidates_block` whitelists fields), the acknowledgement or the student; it is kept in `cycle["grader_info"]` (`compose.question_grader_info`) for the follow-up only and never reported. Every use is fenced `<<<REFERENCIA … REFERENCIA>>>` and governed by `common.reference_rule(kind, lang)`, appended in code like `json_contract` (prompt sha computed before both). **Scoring:** `CycleScore.verdicts` comes before `components` — `correct|partial|incorrect` + a teacher-only `reason` per question with a reference, used as ground truth for the rubric; `verdicts_from` adds `no_reference` in code and raises (retry → `scoring_failed`) when a referenced question has none; a blank cycle is `incorrect` without a call, a failed transcription has no verdict; `verdicts` go to the backend only in `publish`. **Feedback:** `per_question_feedback` (comment only) is crossed with the real questions in code, which attaches `part`/`verdict`, renders «Por pregunta» / «By question» (no part label when the cycle has one question) and sends `perQuestionFeedback`. `copied_span` (6+ consecutive folded words of any reference in the whole text) → one retry with a warning → rewritten without references, log `pablo_feedback_reference_leak` + `pablo_feedback_reference_leaks_total`. **Follow-up:** the references reach `plan_for` (rule only when there are any) and the leak guard treats them like the rubric. **Class report:** `question_analysis` carries `reference` and `verdicts` counts.
- **Questions per cycle**: `next-cycle` sends `questions_per_cycle` (1–3, the teacher's); `compose.validated_parts(result, candidates, k)` takes up to `k` and completes verbatim; the rule is `compose.count_rule` in code (an older backend without it keeps 2–3). `compose.render` drops «Parte A» with one question; `topic_name` strips an outcome's final punctuation; headers are label-like («Tema: …», «Segundo tema de tres: …»); an ALL-CAPS first name is title-cased.
- **Language**: Pablo's fixed messages follow the activity's language (`assessment_turn.language`, sent by the backend), not the tenant's (`texts.lang_of`).
- Prompt keys `ASSESSMENT_{COMPOSE,ACK,FOLLOWUP,SCORE,FEEDBACK,CLASS_REPORT,IMPROVE}` (`src/agents/pablo/prompts.py`), to mirror in the backend's `DEFAULT_PROMPTS`; usage rows carry prompt provenance and `latency_ms`. Tests: `tests/test_pablo_graph.py`, `tests/test_pablo_jobs.py`.

### Key State (`src/graphs/state.py`)

`AgentState` is a `TypedDict` persisted per `thread_id`:
- `messages`: full conversation history (LangGraph `add_messages` reducer)
- `intent`: current classified intent (`sales | tracking | complaint | faq`)
- `structured_intent`: plain JSON dict (`StructuredIntent.model_dump(mode="json")`) — checkpoints must never carry custom Python types. Legacy checkpoints holding Pydantic instances are allow-listed via `JsonPlusSerializer(allowed_msgpack_modules=...)` in `main.py`; readers (`route_from_triage`, `handoff`) handle both shapes.
- `product_catalog`: list of `{product_id, name, description, price, stock}` from NestJS
- `user_context`: dict with `{name, email, phone, address}` from NestJS
- Sales: `sales_step` (0-3), `order_data` (dict), `sales_complete`, `order_confirmed`
- Tracking: `tracking_data` (dict), `tracking_complete`
- Complaint: `complaint_data` (dict), `complaint_complete`
- `execute_confirmed`: boolean flag preventing re-execution

### LLM Client (`src/llm.py`)

`resolve_api_key(config)` extracts the key from `config["configurable"]["openai_api_key"]` (set per-request) or falls back to `settings.openai_api_key`. Each agent node calls this independently — there is no shared client instance.

**Which model: chat vs background.** Chat turns get `llm_model` from the backend, which resolves the tenant's current settings every turn. The non-chat paths fetch credentials themselves and run on the platform's cheap model: `providers/registry.background_model(creds)` reads `backgroundModel` from `GET /internal/agent/credentials`, falling back to `model` for an older backend. That covers `/prospecting/run` (aurora) and `/enrichment/run` (sherlock), whose whole graph uses it. clara carries both — `llm_background_model` in `configurable`, read by `resolve_background_model` — and uses the cheap one only for triage and task extraction (`_structured_call(..., background=True)`); the reply and follow-up drafts stay on the tenant's model because a customer reads them. ismael is the exception among chat agents: its whole chat turn (triage, Moodle support, FAQ answer, teacher-flow classifier and draft) runs on the background model, which `/chat/stream` now puts in `configurable` as `llm_background_model` (taken from the same credentials response); `gemini-3.5-flash` took 10–32 s per short call on 2026-10-06 while `gemini-3.1-flash-lite` answered each turn in ~3 s with the same quality on an eval. Its background RAG job's general answer keeps the tenant's model (`rag_job.job_llm_config` does not copy `llm_background_model`). Every ismael call is also bounded (`common.bounded_structured`: timeout + thinking cap).

### Authentication (`src/auth/service_auth.py`)

Every caller is another service — the NestJS backend and its schedulers — never an end user. There is no browser or mobile client on the other end, which is why this verifies *workload* identity rather than user identity. It replaced the AWS Cognito verifier, whose JWT branch was dead on every production request.

Two accepted credentials, both handled by the single `verify_service_caller` dependency (all four authenticated routes share it):

1. **Google OIDC ID token** on `Authorization: Bearer`. RS256 against Google's JWKS (`https://www.googleapis.com/oauth2/v3/certs`, cached 5 min), issuer `https://accounts.google.com`, `aud` pinned to `SERVICE_AUTH_AUDIENCE` (this service's own Cloud Run URL), then `email_verified == true` and `email` in `SERVICE_AUTH_ALLOWED_SERVICE_ACCOUNTS`. Fails closed when the audience is set but the allowlist is empty. Cloud Run's IAM layer already validates this at the edge (`allow_unauthenticated = false`) and forwards it, so verifying here is defence in depth — which matters because the same image runs under `docker compose` with no Cloud Run in front.
2. **Shared secret** (`WEBHOOK_API_KEY`) on `X-System-Key` or `x-agent-key`, gated by `ALLOW_SHARED_SECRET_AUTH`. A *present but wrong* key does not short-circuit: during the migration the backend sends both credentials, and a stale secret must not veto a valid token.

`get_current_user` always returns the literal `"system"`. **That value is persisted data, not an auth detail** — it is the second segment of every LangGraph thread id in Postgres (`{tenantId}:system:{conversationId}:{codeName}:v{n}`), so returning a service-account email or a numeric `sub` instead would orphan every live conversation's checkpoint history. `tests/test_service_auth.py` guards it.

Rejections funnel through one helper that increments `service_auth_rejects_total{reason}` and logs `service_auth_reject`; the detail returned to the caller stays coarse. API keys from the request body are stripped before emitting `node_update` SSE events.

### Database (`src/db/postgres.py`, `migrations/init.sql`)

asyncpg pool + LangGraph's `AsyncPostgresSaver`. Schema runs idempotently on startup. Tables: `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` (LangGraph), `documents` (pgvector for RAG), `approval_requests` (interrupt audit log).

### Observability (`src/observability.py`)

- Prometheus metrics at `GET /metrics` — request counts, node invocations, order/tracking/complaint funnel, errors
- Optional Langfuse LLM tracing — gracefully disabled if keys not configured

## ⚠️ `agent-migrate` (Cloud Run Job) is broken — known, not yet fixed

Seen 2026-09-29 against `vervux-platform-prod`: its last three executions all
failed (`agent-migrate-4sqmg`, `-gtvsz` on 2026-09-24, `-2b597` on 2026-09-29),
and `yorch-gcp-platform/scripts/migrate.sh yorchio` runs it after every backend
migration, so every such run ends with a failed second job.

Why, from the job's own logs:
- **It starts the server, not a migration.** `envs/prod/jobs.tf` (`module
  "agent_migrate"`) sets no `command`/`args` — unlike `rocky-migrate` /
  `yorchio-migrate` (`npx prisma migrate deploy`) — so it runs this image's
  `CMD`, which is `uvicorn src.main:app`.
- **It dies at config validation first:** the job only gets `DATABASE_URL`, and
  `Settings._require_serper_api_key` refuses to boot without `SERPER_API_KEY`
  (`Value error, SERPER_API_KEY is not set`), `exit(1)`.
- Even with the key it would be wrong: uvicorn never exits, so the job would run
  until its timeout.

Why nothing breaks: this service applies its own schema at startup —
`lifespan` → `src/db/postgres.py::run_migrations` (`migrations/init.sql`,
statement by statement) plus the checkpointer/store `setup()`. The job adds
nothing a deploy does not already do.

One statement also fails at every startup, logged as `migration_statement_failed`:
`CREATE INDEX IF NOT EXISTS approval_requests_thread_status_idx` → `must be owner
of table approval_requests`. The runtime role is not the table owner (the
migrator is), so that index is never created by the service. Harmless today;
running the migration as the migrator is what would fix it.

To fix (pick one): give the job a real one-shot entrypoint (e.g. a small
`python -m src.db.migrate` that opens the pool, calls `run_migrations` and
exits) plus the env `Settings` requires, or delete the job and the
`agent-migrate` step from `migrate.sh`.

## Environment

Copy `.env.example` to `.env`. Required vars:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `SERPER_API_KEY` | Web search for the prospecting agent — a startup validator refuses to boot without it. It checks PRESENCE only: a set-but-invalid key boots fine and then gets `403 Unauthorized` on every `/search` and `/places` call, which `_serper_post` now turns into a fatal `SerperAuthError` so the run ends FAILED instead of reporting `found: 0` (as it silently did before 2026-08-18) |

Optional: `SERVICE_AUTH_AUDIENCE` / `SERVICE_AUTH_ALLOWED_SERVICE_ACCOUNTS` / `ALLOW_SHARED_SECRET_AUTH` (see Authentication — leave the audience empty locally), `WEBHOOK_API_KEY`, `NESTJS_BASE_URL`, `OPENAI_API_KEY` (fallback; per-request key preferred), `GEMINI_*` (absent, the Gemini provider falls back to Application Default Credentials — which is the only auth Agent Platform accepts; it rejects API keys), `LANGFUSE_*`.

## Testing

Tests use `MemorySaver` (no database needed). `conftest.py` sets dummy env vars (`DATABASE_URL`, `SERPER_API_KEY`) and an empty `SERVICE_AUTH_AUDIENCE` before any app modules import, so the suite runs against the shared-secret path without reaching Google's JWKS endpoint. `asyncio_mode = "auto"` in `pyproject.toml` — no need to mark individual tests with `@pytest.mark.asyncio`.

Key test patterns: mock individual agent nodes (e.g., `patch("src.graphs.main_graph.triage_node")`), build a fresh graph with `build_graph(MemorySaver())`, stream with `graph.astream()`, and inspect chunks. Tests verify routing logic, API key security (no leaks into state/interrupts), and auto-chaining flows.

## Studio Graph

`src/graphs/studio_graph.py` is the LangGraph Studio entrypoint — uses an in-memory checkpointer and is not used in production.
