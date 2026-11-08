"""Valid OOXML (.xlsx) workbook generator with entity traceability, separate 'IA · …' sheets, config & spend tabs."""

from __future__ import annotations

import base64
import io
import xml.etree.ElementTree as ET
import zipfile
from typing import Any
from xml.sax.saxutils import escape


def _col_letter(col_idx_1based: int) -> str:
    letters = ""
    n = col_idx_1based
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _normalize_ooxml_text(val: Any) -> str:
    """Normalizes leading and trailing spaces and control characters for valid OOXML (§8)."""
    cleaned = str(val if val is not None else "sin dato").strip()
    # Strip invalid XML 1.0 control characters
    return "".join(ch for ch in cleaned if ch in ("\t", "\n", "\r") or ord(ch) >= 32)


def _build_worksheet_xml(rows: list[list[Any]]) -> bytes:
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "  <sheetData>",
    ]
    for r_idx, row in enumerate(rows, start=1):
        lines.append(f'    <row r="{r_idx}">')
        for c_idx, cell_val in enumerate(row, start=1):
            ref = f"{_col_letter(c_idx)}{r_idx}"
            if isinstance(cell_val, (int, float)) and not isinstance(cell_val, bool):
                lines.append(f'      <c r="{ref}" t="n"><v>{cell_val}</v></c>')
            else:
                text = _normalize_ooxml_text(cell_val)
                escaped = escape(text)
                lines.append(
                    f'      <c r="{ref}" t="inlineStr"><is><t>{escaped}</t></is></c>'
                )
        lines.append("    </row>")
    lines.append("  </sheetData>")
    lines.append("</worksheet>")
    return "\n".join(lines).encode("utf-8")


def build_excel_workbook_bytes(
    report_model: dict[str, Any],
    state: dict[str, Any],
) -> bytes:
    """Generates a valid OOXML (.xlsx) workbook with entity tabs, 'IA · …' tabs, Configuracion, and Gasto_Presupuesto."""
    sections = report_model.get("sections") or {}
    accounts = (sections.get("segmentsAndAccounts") or {}).get("accounts") or []
    zone_cards = sections.get("zoneCountryCards") or []
    customers = (sections.get("currentCustomersByRegion") or {}).get("customers") or []
    generated_at = report_model.get("generatedAt", "sin dato")
    cfg = state.get("config") or {}
    top_k_profiles = list(state.get("top_k_profiles") or [])

    # Sheet 1: Cuentas_Evaluadas (with row-level traceability)
    sheet_accounts: list[list[Any]] = [
        [
            "EntityId",
            "Nombre",
            "ID_Fiscal",
            "Segmento",
            "Puntaje_Determinista",
            "Banda_Prioridad",
            "Pais",
            "Ciudad",
            "Zona",
            "Sitio_Web",
            "Telefono",
            "Email",
            "Fuente",
            "Fecha",
            "URL",
            "Estado_Verificacion",
        ]
    ]
    for acc in accounts:
        website = _normalize_ooxml_text(acc.get("website") or "sin dato")
        tax_id = _normalize_ooxml_text(acc.get("taxId") or "sin dato")
        verified = (
            "verificado"
            if (website not in ("", "sin dato") or tax_id not in ("", "sin dato", "reservado-ley-1581"))
            else "por verificar"
        )
        sheet_accounts.append(
            [
                acc.get("entityId", "sin dato"),
                acc.get("name", "sin dato"),
                tax_id,
                acc.get("segmentCode", "sin dato"),
                acc.get("score", "sin dato"),
                acc.get("priorityBand", "sin dato"),
                acc.get("country", "sin dato"),
                acc.get("city", "sin dato"),
                acc.get("zoneCode", "sin dato"),
                website,
                acc.get("phone", "sin dato"),
                acc.get("email", "sin dato"),
                acc.get("source", "STUDY_PIPELINE"),
                acc.get("fetchedAt", generated_at),
                website if website != "sin dato" else "sin dato",
                verified,
            ]
        )

    # Sheet 2: Zonas_Paises (with row-level traceability)
    sheet_zones: list[list[Any]] = [
        [
            "Codigo",
            "Nombre",
            "Puntaje",
            "Decision",
            "Cobertura_Pct",
            "Dimensiones_Sin_Dato",
            "Fuente",
            "Fecha",
            "URL",
            "Estado_Verificacion",
        ]
    ]
    for zc in zone_cards:
        missing = ", ".join(zc.get("missingDimensions") or []) or "ninguna"
        sheet_zones.append(
            [
                zc.get("code", "sin dato"),
                zc.get("name", "sin dato"),
                zc.get("score", "sin dato"),
                zc.get("decision", "sin dato"),
                zc.get("coveragePct", 100.0),
                missing,
                "RUBRICA_DETERMINISTA",
                generated_at,
                "internal://caleb/zone-rubric",
                "verificado" if missing == "ninguna" else "por verificar",
            ]
        )

    # Sheet 3: Cartera_Clientes (RFM / ABC)
    sheet_customers: list[list[Any]] = [
        [
            "CustomerId",
            "Nombre",
            "Tier_ABC",
            "Estado_Inactividad",
            "Ingresos_Totales",
            "Ordenes",
            "Fuente",
            "Fecha",
            "URL",
            "Estado_Verificacion",
        ]
    ]
    for cust in customers:
        sheet_customers.append(
            [
                cust.get("customerId", "sin dato"),
                cust.get("name", "sin dato"),
                cust.get("abcTier", "sin dato"),
                cust.get("inactivityStatus", "sin dato"),
                cust.get("totalRevenue", "sin dato"),
                cust.get("orderCount", "sin dato"),
                cust.get("source", "INTERNAL_CRM_ERP"),
                generated_at,
                "internal://one/customers",
                "verificado",
            ]
        )

    # Sheet 4: IA · Segunda Opinion (Strictly separated from deterministic scores)
    sheet_ai_opinion: list[list[Any]] = [
        [
            "EntityId",
            "Nombre",
            "Segmento_Determinista",
            "Puntaje_Determinista",
            "Razonamiento_IA",
            "Estado_Evidencia_IA",
            "Estado_Verificacion",
        ]
    ]
    for acc in accounts:
        sheet_ai_opinion.append(
            [
                acc.get("entityId", "sin dato"),
                acc.get("name", "sin dato"),
                acc.get("segmentCode", "sin dato"),
                acc.get("score", "sin dato"),
                acc.get("aiRationale", "Segunda opinión generada sin alterar puntaje determinista."),
                acc.get("aiProfileStatus", "con evidencia"),
                "por verificar",
            ]
        )

    # Sheet 5: IA · Perfiles Top K
    sheet_ai_topk: list[list[Any]] = [
        [
            "EntityId",
            "Nombre",
            "Banda",
            "Resumen_Ejecutivo_IA",
            "Abordaje_Sugerido_IA",
            "Estado_Verificacion",
        ]
    ]
    for prof in top_k_profiles:
        sheet_ai_topk.append(
            [
                prof.get("entityId", "sin dato"),
                prof.get("name", "sin dato"),
                prof.get("priorityBand", "sin dato"),
                prof.get("executiveSummary", "sin dato"),
                prof.get("suggestedApproach", "sin dato"),
                "por verificar",
            ]
        )

    # Sheet 6: Configuracion
    sheet_config: list[list[Any]] = [
        ["Parametro", "Valor"],
        ["schemaVersion", cfg.get("schemaVersion", 1)],
        ["studyTitle", cfg.get("studyTitle", "sin dato")],
        ["tier", cfg.get("tier", "PILOT")],
        ["modules", ", ".join(cfg.get("modules") or [])],
        ["companyName", (cfg.get("businessProfile") or {}).get("companyName", "sin dato")],
        ["industry", (cfg.get("businessProfile") or {}).get("industry", "sin dato")],
    ]

    # Sheet 7: Gasto_Presupuesto
    limits = sections.get("limitsAndGlossary") or {}
    sheet_spend: list[list[Any]] = [
        ["Concepto", "Monto_USD", "Detalle"],
        ["Tope_Presupuesto_Interno", limits.get("budgetCapUsd", 0.0), "Precio * (1 - margen objetivo)"],
        ["Gasto_Liquidado_Real", limits.get("settledSpendUsd", 0.0), "Total acumulado en la corrida"],
    ]
    for deg in limits.get("degradations") or []:
        sheet_spend.append(
            [
                f"Degradacion:{deg.get('level', 'DEGRADED')}",
                0.0,
                deg.get("reason", "sin dato"),
            ]
        )

    sheets_spec: list[tuple[str, list[list[Any]]]] = [
        ("Cuentas_Evaluadas", sheet_accounts),
        ("Zonas_Paises", sheet_zones),
        ("Cartera_Clientes", sheet_customers),
        ("IA · Segunda Opinion", sheet_ai_opinion),
        ("IA · Perfiles Top K", sheet_ai_topk),
        ("Configuracion", sheet_config),
        ("Gasto_Presupuesto", sheet_spend),
    ]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types].xml
        ct_overrides = "\n".join(
            f'  <Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for i in range(1, len(sheets_spec) + 1)
        )
        content_types_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
{ct_overrides}
</Types>"""
        zf.writestr("[Content_Types].xml", content_types_xml.encode("utf-8"))

        # _rels/.rels
        root_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
        zf.writestr("_rels/.rels", root_rels_xml.encode("utf-8"))

        # xl/workbook.xml
        sheet_entries = "\n".join(
            f'    <sheet name="{escape(_normalize_ooxml_text(name))}" sheetId="{i}" r:id="rId{i}"/>'
            for i, (name, _) in enumerate(sheets_spec, start=1)
        )
        workbook_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
{sheet_entries}
  </sheets>
</workbook>"""
        zf.writestr("xl/workbook.xml", workbook_xml.encode("utf-8"))

        # xl/_rels/workbook.xml.rels
        wb_rels_entries = "\n".join(
            f'  <Relationship Id="rId{i}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(sheets_spec) + 1)
        )
        wb_rels_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{wb_rels_entries}
</Relationships>"""
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels_xml.encode("utf-8"))

        # xl/worksheets/sheet{i}.xml
        for i, (_, rows) in enumerate(sheets_spec, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", _build_worksheet_xml(rows))

    return buf.getvalue()


def build_excel_workbook_base64(
    report_model: dict[str, Any],
    state: dict[str, Any],
) -> str:
    """Returns base64-encoded valid OOXML (.xlsx) bytes."""
    return base64.b64encode(build_excel_workbook_bytes(report_model, state)).decode("ascii")


def parse_xlsx_workbook_sheets(xlsx_bytes: bytes) -> dict[str, list[list[Any]]]:
    """Parses our OOXML (.xlsx) archive into `{sheet_name: [[cell_values, ...], ...]}` for verification."""
    ns_main = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ns_rel = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    ns_pkg_rel = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}

    out: dict[str, list[list[Any]]] = {}
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes), mode="r") as zf:
        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map: dict[str, str] = {}
        for rel in rels_root.findall("pr:Relationship", ns_pkg_rel):
            rel_map[rel.attrib["Id"]] = f"xl/{rel.attrib['Target']}"

        for sheet_el in wb_root.findall("s:sheets/s:sheet", ns_main):
            name = sheet_el.attrib["name"]
            r_id = sheet_el.attrib[f"{{{ns_rel['r']}}}id"]
            target_path = rel_map[r_id]
            ws_root = ET.fromstring(zf.read(target_path))
            rows_parsed: list[list[Any]] = []
            for row_el in ws_root.findall("s:sheetData/s:row", ns_main):
                row_vals: list[Any] = []
                for cell_el in row_el.findall("s:c", ns_main):
                    cell_type = cell_el.attrib.get("t", "n")
                    if cell_type == "inlineStr":
                        t_el = cell_el.find("s:is/s:t", ns_main)
                        row_vals.append(t_el.text if t_el is not None and t_el.text is not None else "")
                    else:
                        v_el = cell_el.find("s:v", ns_main)
                        raw_v = v_el.text if v_el is not None and v_el.text is not None else "0"
                        row_vals.append(float(raw_v) if "." in raw_v else int(raw_v))
                rows_parsed.append(row_vals)
            out[name] = rows_parsed
    return out
