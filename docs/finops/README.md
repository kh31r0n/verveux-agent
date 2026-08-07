# FinOps spec corpus

Design and operational documents for the FinOps / IT cost-management initiative,
moved here on 2026-08-07 from the `helena` workspace root when FinOps left the
Helena stack.

**Read these as historical record, not as a description of the current system.**
They were written against `helena-backend` / `helena-frontend` / `helena-agent`,
and every path they cite has since moved:

| Then | Now |
|---|---|
| `helena-backend/src/finops/*` | `verveux-backend/src/finops/*` (tenant-scoped) |
| `helena-agent/src/spend_fix.py` | `verveux-agent/src/finops/spend_fix.py` |
| `helena-agent/src/auth/api_key.py` | `verveux-agent/src/auth/machine.py` |
| 19 `finops_*` tables in helena-backend | dropped — `20260805000000_remove_finops` |

The largest behavioural difference: these documents describe an **async**
`upload → S3 → Airflow → agent-fix → batch-insert` pipeline. `verveux-backend`'s
`SpendImportService` currently parses and inserts **in-request**, with no Airflow
DAG. `POST /agent/fix-spend-file` was ported for the async path but has no caller
yet.

| File | What it is |
|---|---|
| `it-cost-center_charter.en.md` | The initiative charter, v1.0.0 (2026-07-02) |
| `GAP_charter_vs_code.md` | Gap report: where the charter lagged the code (2026-07-21) |
| `PLAN_REALSPEND_IMPORT_WORKFLOW.md` | Design for the async import workflow (the big one) |
| `OPS_REALSPEND_IMPORT.md` | Operator runbook for that workflow |
| `PRS_REALSPEND_IMPORT.md` | Per-repo PR descriptions for the same work |
| `FINOPS_DATA_ARCHITECTURE.html` | Data-architecture reference (rendered) |

A happy-path sample workbook lives at `tests/fixtures/realspend-test-happy-usd.xlsx`.
