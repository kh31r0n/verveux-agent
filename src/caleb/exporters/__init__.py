"""Deliverable generators (JSON Report Model v1, Markdown, CSV, OOXML Excel, One CRM & Aurora proposals) for Caleb (Phase 5)."""

from .crm_and_aurora import build_aurora_proposals, build_crm_export_payload
from .csv_exporter import export_entities_csv
from .excel_workbook import (
    build_excel_workbook_base64,
    build_excel_workbook_bytes,
    parse_xlsx_workbook_sheets,
)
from .markdown_report import render_markdown_report
from .report_model import (
    build_caleb_report_model_v1,
    compute_deterministic_sensitivity,
    parse_mixed_locale_number,
    sanitize_natural_person_entity,
    verify_prose_numbers_against_model,
)

__all__ = [
    "build_aurora_proposals",
    "build_caleb_report_model_v1",
    "build_crm_export_payload",
    "build_excel_workbook_base64",
    "build_excel_workbook_bytes",
    "compute_deterministic_sensitivity",
    "export_entities_csv",
    "parse_mixed_locale_number",
    "parse_xlsx_workbook_sheets",
    "render_markdown_report",
    "sanitize_natural_person_entity",
    "verify_prose_numbers_against_model",
]
