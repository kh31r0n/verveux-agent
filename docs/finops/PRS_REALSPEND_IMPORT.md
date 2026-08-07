# Pull-request descriptions — RealSpendRecord async import workflow

> **Superseded (2026-08-07).** FinOps has left the Helena stack: the module now
> lives in `verveux-backend/src/finops/` and the spend-file repair in
> `verveux-agent/src/finops/`. The `helena-*` PRs below are historical, and the
> async S3 → Airflow path they describe was not carried over — verveux imports
> in-request. See `README.md` in this directory.

Four PRs (one per repo) for the upload → S3 → Airflow → agent-fix → batch-insert
pipeline. Design: `PLAN_REALSPEND_IMPORT_WORKFLOW.md`; deploy: `OPS_REALSPEND_IMPORT.md`.

> **Branching note:** in each repo these changes currently sit uncommitted on an
> unrelated feature branch (backend `func/organizationStep`, frontend
> `func/automationCatalog`, agent `func/organizationConfirmation`, airflow `main`),
> mixed with a broader uncommitted FinOps body of work. Recommend cutting a fresh
> `feat/realspend-async-import` branch per repo and staging only the files listed
> below so each PR is reviewable in isolation.

---

## PR 1 — helena-backend: async spend-import (tracker + transactional batch insert)

**Summary**
Replaces the in-request Excel parse+insert with an async pipeline: the upload
endpoint stores the file in S3, creates an `ExecutionTracker` + `SpendImportBatch`,
and triggers an Airflow DAG. A new machine endpoint accepts pre-parsed rows and
inserts them transactionally (all-or-nothing), returning per-row failures the DAG
feeds to the repair agent. A tracker callback drives UI progress.

**Changes**
- Prisma: `ExecutionTrackerStatus` enum, `ExecutionTracker` (1:1 → `SpendImportBatch`),
  `SpendBatchAttempt` (idempotency ledger); migration `20260625000000_add_execution_tracker`.
- `finops/execution-tracker/` module: service (create / terminal-guarded update /
  findOne / findByRequestId), `GET` (human, `VIEW_FINOPS`) + `PATCH` (machine,
  `AirflowApiKeyGuard`) at `/finops/spend/execution-trackers/:id`.
- `real-spend`: `RealSpendService.createBatch()` (idempotency replay → pre-validation →
  single `$transaction`, per-row failure mapping, max-rows cap); `POST /finops/spend/batch`
  (`@Public()` + `AirflowApiKeyGuard`, raw machine JSON); `SpendImportService.enqueueImport()`;
  `POST /finops/spend/import` now async; imports `AirflowModule` + `ExecutionTrackerModule`.
- `SpendBatchRecordDto` is intentionally **permissive** (semantic validation in the
  service) so a malformed row yields per-row failures instead of a 400 under the
  global `forbidNonWhitelisted` pipe.
- Config/env: `FINOPS_IMPORT_DAG_ID`, `FINOPS_IMPORT_MAX_AGENT_RETRIES`,
  `FINOPS_IMPORT_MAX_ROWS`; reuses `AIRFLOW_WEBHOOK_API_KEY`/`_HMAC_SECRET`.
- Audit: added `BATCH_CREATE` action.

**Testing** — `npx jest` 84/84 (createBatch happy/rollback/idempotency/dup/cap/
permissive-row; tracker terminal-guard; **real-`ValidationPipe` integration spec**;
enqueue happy/dedup/S3-unconfigured/trigger-failure). `tsc --noEmit` clean.

**Deploy** — `npm run prisma:migrate:deploy`; set `FINOPS_S3_BUCKET`, `AIRFLOW_*`,
`FINOPS_IMPORT_*`. See `OPS_REALSPEND_IMPORT.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## PR 2 — aiops-airflow: finops_spend_import DAG

**Summary**
New backend-triggered DAG that downloads the uploaded file from S3, parses it,
inserts rows via the backend batch endpoint, and—on failure—loops through the
helena-agent repair service up to `maxAgentRetries`, updating the ExecutionTracker
at each step. Backend-internal: **not** `agent-enabled`.

**Changes**
- `dags/finops_spend_import.py` — Airflow 3 SDK DAG (`extract_conf` →
  `process_spend_file` single-task loop → `notify_backend` `ALL_DONE`). The
  whole loop is one task because KubernetesExecutor pods don't share `/tmp`.
  Enforces the max-rows cap pre-insert; the terminal task also PATCHes the
  tracker FAILED if the loop crashed (webhook only covers `AutomationRequest`).
- `dags/utils/finops.py` — connection resolution, signed POST/PATCH (x-api-key +
  optional HMAC), agent call, S3 download, parse + alias-normalize, `decide_outcome`.
  pandas/airflow imports are lazy so the pure logic is testable.
- `dags/utils/finops_test.py` — 6 tests for the pure helpers.

**Testing** — `PYTHONPATH=dags python3 dags/utils/finops_test.py` 6/6;
`py_compile` clean. (DAG import-check requires the Airflow image.)

**Deploy** — sync the files; create the **`helena_agent`** HTTP connection; confirm
`aws_default` + `helena_backend`; IRSA S3 read on the task pods. Do **not** enable
it in the agent `WorkflowCatalog`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## PR 3 — helena-agent: /agent/fix-spend-file repair capability

**Summary**
New machine-to-machine endpoint that repairs a malformed spend file in S3: reads
the original, maps columns toward the canonical schema (deterministic alias map +
an aggressive LLM column-mapping pass), coerces values, writes a corrected file
back to S3, and reports rows it couldn't fix. Non-SSE JSON, API-key (+ optional
HMAC) auth — no user JWT, no interrupt/approval.

**Changes**
- `src/main.py` — `POST /agent/fix-spend-file` (`Depends(require_machine_api_key)`).
- `src/auth/api_key.py` — API-key guard + `verify_hmac` (optional HMAC over the body).
- `src/integrations/s3.py` — lazy-`boto3` read/write (IRSA).
- `src/spend_fix.py` — coercion (EU/US amount separators, multi-format dates→ISO,
  currency, paymentOrigin synonyms→enum), alias + LLM mapping, `fix_spend_file`.
  Aggressive defaults for non-identifying fields; identifying fields never fabricated;
  unfixable rows kept (honest all-or-nothing).
- `src/config.py` — `agent_machine_api_key`, `agent_machine_hmac_secret`,
  `finops_s3_bucket/region`, `spend_fix_model`; **`extra="ignore"`** (pre-existing fix:
  the dev `.env` carried keys the model forbade, breaking the whole test suite).
- `pyproject.toml` — `boto3`, `pandas`, `openpyxl`.

**Testing** — `pytest` 40/40 (coercions, mapping, find_unfixable, fix_spend_file with
I/O+LLM patched, api-key guard 401/403/503/ok, HMAC). **Run `uv lock && uv sync`** for
the new deps (lazy-imported, so tests pass without them).

**Deploy** — set `AGENT_MACHINE_API_KEY`, `FINOPS_S3_BUCKET/REGION` (+ optional
`AGENT_MACHINE_HMAC_SECRET`); IRSA S3 read+write on the agent pod.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## PR 4 — helena-frontend: async import progress UI

**Summary**
Switches the FinOps Real-spend Import tab to the async flow: upload returns a
tracker, and a new component polls it, showing live status (parsing → inserting →
agent-fixing → success/failed), inserted/total, retry rounds, and failures.

**Changes**
- `types/finops.ts` — `ExecutionTrackerStatus`, `TERMINAL_TRACKER_STATUSES`,
  `EnqueuedImport`, `ExecutionTracker`.
- `services/finops/real-spend.service.ts` — `import()` → `EnqueuedImport`; `getTracker(id)`.
- `stores/finops/realSpend.ts` — `getTracker` action + `currentTracker` (poll doesn't
  toggle the shared loading flag).
- `components/finops/real-spend/SpendImportProgress.vue` (new) — 2 s REST poll,
  stops on terminal/unmount, guards overlapping requests, "view batch" on success.
- `SpendImportUpload.vue` — async submit, emits `enqueued`. `RealSpendPanel.vue` — wiring.
- `utils/finopsConstants.ts` — `EXECUTION_TRACKER_STATUS_COLORS`.
- `i18n/locales/{es,pt,en}.ts` — `executionTrackerStatus` + `import.progress.*`
  (identical key structure across all three).

**Testing** — `npx vue-tsc --noEmit` clean; i18n key parity verified. Full build
needs `sudo npm install && npm run build` (root-owned node_modules).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
