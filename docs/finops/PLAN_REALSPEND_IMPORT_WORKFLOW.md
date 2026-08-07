# Plan — Async `RealSpendRecord` File-Import Workflow (S3 → Airflow → Agent → Backend)

> Cross-repo feature spanning **helena-backend** (NestJS/Prisma), **aiops-airflow** (Airflow DAGs, K8s), **helena-agent** (FastAPI/LangGraph), and **helena-frontend** (Vue 3). Entry point in the UI is `/helena/finops` → **Real spend** tab → **Import**.

---

## 0. Decisions locked (from review)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Where the file is parsed | **DAG parses** (downloads from S3, parses XLSX/CSV → normalized rows) and POSTs rows to a new backend **batch-insert** endpoint that only validates + inserts. |
| D2 | How the agent handles the file | **Agent reads & writes S3** — a new machine endpoint downloads the original from S3, repairs it, writes a corrected file back to S3, returns `correctedS3Key`. |
| D3 | ExecutionTracker modeling | **New dedicated `ExecutionTracker` model**, 1:1 FK to `SpendImportBatch`. Tracker owns cross-system orchestration state; batch keeps row-level counts. |
| D4 | Retry loop + failure semantics | **Max 3 agent-fix retries** (configurable), **all-or-nothing**: each insert attempt is a single DB transaction; on any row failure the whole attempt rolls back; after retries are exhausted **nothing persists** and the tracker ends `FAILED`. |

### Confirmed decisions for the secondary questions
- **File fetch by DAG**: pass the raw `s3Key` in `dag_run.conf`; the DAG reads via `S3Hook(aws_conn_id="aws_default")` and the agent reads/writes via `boto3`, both using **IRSA** (pod IAM role). No presigned URLs (avoids TTL expiry on long agent runs).
- **Auth on machine endpoints**: reuse the existing **`AirflowApiKeyGuard`** pattern (`x-api-key` + optional `x-hmac-signature` HMAC-SHA256) for DAG→backend calls; a new API-key guard in the agent for DAG→agent. 
- **Result callback**: granular mid-pipeline status goes to a **new finops tracker-callback endpoint**; the existing `POST /api/v3/airflow-execution/webhook` is used as the DAG's terminal `notify_backend` safety net. 
- **Frontend live updates**: **REST polling** of the tracker every ~2 s while non-terminal (defer a `/finops` WebSocket namespace). 
- **`ingestionChannel`** for these records: `EXCEL` (file-based). *(Not yet confirmed — revisit if a distinct `AIRFLOW`/`API` channel is wanted.)*
- **Idempotency key** for batch insert: `idempotencyKey = "${batchId}:${attempt}"`.
---

## 1. End-to-end flow (11 steps → components)

```
            helena-frontend                helena-backend                      aiops-airflow (K8s)            helena-agent
 ┌──────────────────────────┐   ┌────────────────────────────────┐   ┌───────────────────────────┐   ┌────────────────────────┐
 │ SpendImportUpload.vue     │   │ POST /finops/spend/import-jobs │   │  DAG finops_spend_import   │   │ POST /agent/fix-spend-  │
 │  (1) user picks file ─────┼──▶│  (2) create ExecutionTracker   │   │                            │   │      file (machine)     │
 │  (multipart upload)       │   │      + SpendImportBatch        │   │  extract_conf              │   │                         │
 │                           │   │  (3) S3Service.putObject ──────┼─▶ │  ↓                         │   │ (8) boto3: read S3 →    │
 │  poll tracker every ~2s   │   │  (4) AirflowService.trigger ───┼──▶│  process_spend_file (loop):│   │     LLM repair →        │
 │  show progress  ◀─────────┼───┤      DagRun(conf)              │   │   • S3Hook download+parse  │   │     write corrected S3  │
 │                           │   │                                │   │   • (5) POST /spend/batch ─┼──▶│     return correctedS3Key│
 │                           │   │  POST /finops/spend/batch  ◀───┼───┤        (txn all-or-nothing)│   └────────────────────────┘
 │                           │   │   → RealSpendService.create-   │   │   • ok? (6) tracker SUCCESS│              ▲
 │                           │   │       Batch (transactional)    │   │   • fail & attempt<3:      │              │
 │                           │   │  PATCH /spend/imports/:id  ◀───┼───┤     (7) tracker AGENT_FIX ─┼──────────────┘
 │                           │   │   (tracker status callback)    │   │     call agent, (9) re-    │
 │                           │   │                                │   │     parse corrected, loop  │
 │                           │   │  POST /airflow-execution/      │   │   • (10) fail again→loop   │
 │                           │   │   webhook (terminal safety) ◀──┼───┤   • (11) success→SUCCESS / │
 │                           │   │                                │   │     exhausted→FAILED       │
 └──────────────────────────┘   └────────────────────────────────┘   │  notify_backend (ALL_DONE) │
                                                                       └───────────────────────────┘
```

| Step | Component | Artifact (new ⊕ / changed △) |
|------|-----------|------------------------------|
| 1 Upload file | frontend `SpendImportUpload.vue` → new async endpoint | △ point store action at `import-jobs` |
| 2 Create tracker | backend `ExecutionTrackerService.create()` | ⊕ model, migration, service, module |
| 3 Upload to S3 | backend `S3Service.putObject()` (already exists) | △ make mandatory for async path; store `s3Key` |
| 4 Start DAG | backend `AirflowService.triggerDagRun(dagId, conf)` (exists) | ⊕ internal trigger call (bypass agent catalog) |
| 5 Parse + create records | DAG parses → `POST /finops/spend/batch` → `RealSpendService.createBatch()` | ⊕ DAG, ⊕ batch endpoint + DTO + service method |
| 6 On success update tracker | backend tracker callback | ⊕ `PATCH /finops/spend/imports/:trackerId` |
| 7 On insert fail, ask agent | DAG → `POST /agent/fix-spend-file` | ⊕ agent endpoint |
| 8 Agent fixes + writes S3 | agent boto3 + repair logic | ⊕ `src/integrations/s3.py` + fix node |
| 9 Resume inserting | DAG re-parses corrected file → same batch endpoint | (reuse) |
| 10 Fail again → loop | DAG loop bounded by `maxAgentRetries` | ⊕ loop logic + `retryCount` on tracker |
| 11 Success → tracker SUCCESS | backend tracker callback + webhook | (reuse step 6) |

---

## 2. Backend (helena-backend)

### 2.1 Prisma model — `ExecutionTracker`
New `finops_`-prefixed table, 1:1 with `SpendImportBatch`. Add to `prisma/schema.prisma`:

```prisma
enum ExecutionTrackerStatus {
  PENDING        // created, file not yet in S3
  UPLOADING      // backend pushing file to S3
  QUEUED         // DAG triggered, not yet picked up
  PARSING        // DAG downloading + parsing
  INSERTING      // calling /spend/batch
  AGENT_FIXING   // an insert attempt failed; agent repairing the file
  SUCCESS        // all rows inserted (terminal)
  FAILED         // exhausted retries / fatal error (terminal, nothing persisted)
}

model ExecutionTracker {
  id             String                 @id @default(cuid())
  status         ExecutionTrackerStatus @default(PENDING)
  currentStep    String?                // human-readable step label for the UI
  s3Key          String?                // original uploaded file
  correctedS3Key String?                // last agent-corrected file
  dagId          String?
  dagRunId       String?
  requestId      String                 @unique   // correlates Airflow callbacks (mirror AutomationRequest)
  retryCount     Int                    @default(0)
  maxRetries     Int                    @default(3)
  totalRows      Int?
  insertedRows   Int?
  result         Json?                  // success summary
  error          Json?                  // last error / unfixable rows
  metadata       Json?                  // per-step audit trail (append-only)
  importBatchId  String?                @unique
  importBatch    SpendImportBatch?      @relation(fields: [importBatchId], references: [id])
  uploadedByUserId String?
  createdAt      DateTime               @default(now())
  updatedAt      DateTime               @updatedAt

  @@index([status, createdAt])
  @@map("finops_execution_trackers")
}
```
- Add the reverse relation field on `SpendImportBatch` (`executionTracker ExecutionTracker?`).
- Migration: `npm run prisma:migrate` → `prisma/migrations/<ts>_add_execution_tracker`. (Follow the existing `Decimal(18,4)` / `finops_` conventions.)
- **Idempotency store** for batch inserts: small table `finops_spend_batch_attempts { idempotencyKey @id, trackerId, committed Boolean, result Json, createdAt }` (or reuse `metadata`). Guards a DAG-task network retry from double-committing a *successful* attempt.

### 2.2 New module `ExecutionTrackerModule` (`src/finops/execution-tracker/`)
- `execution-tracker.service.ts` — `create()`, `update(id, patch)` (idempotent; never regress past a terminal state — copy the **status-precedence** idea from `airflow-execution.service.ts`), `findOne()`, `list()`.
- `execution-tracker.controller.ts` (mounted at `finops/spend/execution-trackers` — chosen over `imports/:trackerId/tracker` to avoid any ambiguity with the batch-id-keyed `imports/:id` route):
  - `GET /finops/spend/execution-trackers/:id` — **human** read; guard `@RequirePermissions(VIEW_FINOPS)`. Returns `{ data }`.
  - `PATCH /finops/spend/execution-trackers/:id` — **machine** status callback from the DAG; `@Public()` + `AirflowApiKeyGuard`. Body: `{ status?, currentStep?, retryCount?, correctedS3Key?, dagId?, dagRunId?, totalRows?, insertedRows?, result?, error?, metadata? }`.
- Register in `app.module.ts` as a sibling finops module.

### 2.3 Async upload endpoint (steps 1–4)
Add to `real-spend.controller.ts` (or a thin `SpendImportJobsController`):
- `POST /finops/spend/import-jobs` — Multer multipart, `@RequirePermissions(IMPORT_SPEND)` (FinanceRole `IMPORT_OPERATOR`/`FINANCE_ADMIN`).
- Service (`spend-import.service.ts`, new `enqueueImport()`), in order:
  1. Compute SHA-256 checksum (reuse existing dedup against `SpendImportBatch.fileChecksum`).
  2. Create `SpendImportBatch` (status `PENDING`) + `ExecutionTracker` (status `UPLOADING`, fresh `requestId = uuid()`, `maxRetries` from config, `uploadedByUserId`).
  3. `S3Service.putObject("finops/imports/{batchId}/{filename}", buffer, contentType)` — **mandatory** here (today it's best-effort). Store `s3Key` on both batch and tracker; set tracker `QUEUED`.
  4. `AirflowService.triggerDagRun(FINOPS_IMPORT_DAG_ID, conf)` where
     `conf = { requestId, executionTrackerId, batchId, s3Key, fileType, maxAgentRetries, callbackBaseUrl }`.
     Persist returned `dagRunId`/`dagId` on the tracker.
  5. Return `{ trackerId, batchId, requestId, status }` immediately (202-style).
- **Wiring note**: import `AirflowModule` into the finops real-spend module to inject `AirflowService` directly. Do **not** route through `AutomationService`/`WorkflowCatalog` (that gating exists to control the *agent*, not internal pipelines). The DAG must **not** carry the `agent-enabled` tag.

### 2.4 Batch insert endpoint (steps 5/9) — transactional, all-or-nothing
- `POST /finops/spend/batch` — `@Public()` + `AirflowApiKeyGuard`. DTO:
  ```ts
  class SpendBatchDto {
    executionTrackerId: string;
    batchId: string;
    idempotencyKey: string;       // "${batchId}:${attempt}"
    attempt: number;
    records: NormalizedSpendRecordDto[];   // money fields as strings (Decimal)
  }
  ```
- `RealSpendService.createBatch(records, { batchId, transactional: true })`:
  1. **Idempotency check**: if `idempotencyKey` already committed → return its stored result, do not re-insert.
  2. **Pre-validation pass (no DB writes)**: for every row compute fingerprint (`spend.util.ts::computeFingerprint`), freeze FX (`FxModule.freeze`), validate invariants. Collect per-row diagnostics. If **any** row is invalid → return `{ ok: false, failedCount, perRow: [{ rowIndex, status, error }] }` and persist nothing.
  3. **Transactional insert**: `prisma.$transaction` inserting all rows (link `importBatchId`). If a DB constraint trips mid-transaction (e.g. P2002 fingerprint dup against an already-committed record, P2003 bad FK) → the transaction **rolls back**; map the Prisma error back to the offending `rowIndex`; return `{ ok: false, ... }`. Nothing persists.
  4. On full success → commit, record the `idempotencyKey` as committed, return `{ ok: true, insertedCount, perRow: [...] }`.
- Response shape drives the DAG's decision (loop vs done) and the tracker's `error`/`result`.
- **Why all-or-nothing is clean here**: each attempt re-submits the *entire* corrected row set; only the final good attempt commits, so there's never partial/duplicated data between attempts. The `sourceFingerprint` UNIQUE constraint remains defense-in-depth.
- **Risk**: a single large transaction (thousands of rows) can hit statement timeouts — chunk `createMany` *inside* one `$transaction`, and cap batch size (e.g. reject > N rows; DAG should rarely need chunking but document the ceiling).
- **Rows without `invoiceId`/`sourceRef`** produce a `null` fingerprint and are not deduped. For all-or-nothing this is acceptable within a run (rollback prevents dupes), but require at least one stable id per importable row (validate in DTO) so re-runs across separate uploads don't double-count.

### 2.5 Permissions & config
- `finance-role-permissions.map.ts`: human tracker views → `VIEW_FINOPS`; upload → `IMPORT_SPEND`. Machine endpoints are `@Public()` to Keycloak, protected only by `AirflowApiKeyGuard`.
- New env (all via `env.validation.ts` Joi, fail-fast): `FINOPS_IMPORT_DAG_ID` (default `finops_spend_import`), `FINOPS_IMPORT_MAX_AGENT_RETRIES` (default `3`). Reuse the existing `AIRFLOW_WEBHOOK_API_KEY` / `AIRFLOW_WEBHOOK_HMAC_SECRET` for the finops machine endpoints (or add `FINOPS_MACHINE_API_KEY` if separation is desired).
- Follow the repo's `nestjs-best-practices` skill for module/DI/guard structure.

### 2.6 Backend files
- ⊕ `src/finops/execution-tracker/{execution-tracker.module,.service,.controller}.ts` + DTOs
- ⊕ `src/finops/real-spend/dto/spend-batch.dto.ts`
- △ `src/finops/real-spend/real-spend.controller.ts` (+ `import-jobs`, `+ /spend/batch`)
- △ `src/finops/real-spend/real-spend.service.ts` (`createBatch`)
- △ `src/finops/real-spend/spend-import.service.ts` (`enqueueImport`)
- △ `src/finops/real-spend/real-spend.module.ts` (import `AirflowModule`, `ExecutionTrackerModule`)
- △ `prisma/schema.prisma` + new migration
- △ `src/config/{config.service,env.validation}.ts`
- △ `src/app.module.ts` (register `ExecutionTrackerModule`)

---

## 3. Airflow DAG (aiops-airflow)

### 3.1 New file `dags/finops_spend_import.py` (Airflow 3 **SDK** convention)
Module docstring = source of truth (published as `doc_md`): documents `dag_run.conf` fields, connections, variables.

```
schedule=None, max_active_runs=10, tags=["finops", "spend-import"]   # NOT "agent-enabled"
```

**conf** (from backend): `{ requestId, executionTrackerId, batchId, s3Key, fileType, maxAgentRetries, callbackBaseUrl }`.

**Connections**: `aws_default` (S3 read/write via IRSA), `helena_backend` (batch insert + tracker callback + webhook — host/api-key/hmac), and a **new** `helena_agent` HTTP connection (host = agent base URL, password = agent API key).

**Task graph** — kept minimal because KubernetesExecutor pods don't share `/tmp`:
1. `extract_conf` — validate required conf, raise `ValueError` if missing. Returns normalized dict.
2. `process_spend_file` — **single task that owns the whole loop** (one pod, S3 for files, in-memory state):
   - PATCH tracker → `PARSING`; `S3Hook.get_key(s3Key)` → parse with `pandas` (XLSX/CSV) → `rows[]`; PATCH `totalRows`.
   - `attempt = 0`; `current_key = s3Key`.
   - **loop while** `attempt <= maxAgentRetries`:
     - PATCH tracker → `INSERTING`.
     - `POST {backend}/api/v3/finops/spend/batch` with `{executionTrackerId, batchId, idempotencyKey: f"{batchId}:{attempt}", attempt, records: rows}` (`x-api-key` [+ HMAC]).
     - if `ok` → PATCH tracker `SUCCESS` (`insertedRows`, `result`); **break**.
     - else if `attempt == maxAgentRetries` → PATCH tracker `FAILED` (`error` = per-row failures + `unfixableRows`); raise to fail the task.
     - else → PATCH tracker `AGENT_FIXING` (`retryCount = attempt+1`); `POST {agent}/agent/fix-spend-file` with `{s3Key: current_key, errors: failedRows, expectedSchema: "NormalizedSpendRecord"}`; read `{correctedS3Key, fixedRowCount, unfixableRows}`; `current_key = correctedS3Key`; PATCH tracker `correctedS3Key`; re-download + re-parse `current_key` → `rows`; `attempt += 1`.
3. `notify_backend_task` (`trigger_rule=ALL_DONE`) — reuse `utils/webhook.py::notify_backend()` to POST terminal `SUCCESS`/`FAILED` to `/api/v3/airflow-execution/webhook` with `requestId`. **Safety net** if `process_spend_file` crashes before its final tracker PATCH.

`extract_conf >> process_spend_file >> notify_backend_task`

### 3.2 Reuse / add
- Reuse `dags/utils/webhook.py::notify_backend` (and its `helena_backend` connection resolution + HMAC).
- Add a small `dags/utils/finops.py` helper (HTTP POST with `x-api-key`/HMAC to backend & agent, S3 read/write via `S3Hook`, file parsing) — keep secrets out of logs, ISO-8601 UTC timestamps, follow existing logging conventions.
- **Trade-off**: single orchestrating task sacrifices per-step Airflow task visibility, but the **tracker callbacks** give finer UI progress than task states would. (Alternative: multi-task graph with `BranchPythonOperator` + `TriggerDagRunOperator` self-loop — more "Airflow-native" but materially more complex on KubernetesExecutor; not recommended for v1.)

---

## 4. helena-agent

### 4.1 New machine endpoint `POST /agent/fix-spend-file` (`src/main.py`)
- **Non-SSE JSON**, **API-key** authenticated (new guard, NOT Keycloak/SSE; no LangGraph `interrupt()`/approval — this is machine-to-machine).
- Request: `{ s3Key, errors: [{ rowIndex, field?, message }], expectedSchema: "NormalizedSpendRecord", batchId }`.
- Response: `{ correctedS3Key, fixedRowCount, unfixableRows: [{ rowIndex, reason }] }`.
- Behavior: download original from S3 → repair rows toward the canonical `NormalizedSpendRecord` shape → write corrected file to `finops/imports/{batchId}/corrected-{n}.{ext}` → return key. Must always emit a *complete* file (fixed + originally-good rows) so the DAG can re-parse the whole set.

### 4.2 S3 + deps
- ⊕ `src/integrations/s3.py` — `read_object`, `write_object` via `boto3` (IRSA). Add `boto3` with `uv add boto3` (updates `pyproject.toml`/`uv.lock`).
- Env: `AGENT_MACHINE_API_KEY`, `FINOPS_S3_BUCKET`/`FINOPS_S3_REGION` (mirror backend).

### 4.3 Repair logic (the biggest functional unknown — **needs a spec**)
v1 proposal: a deterministic normalizer (header mapping, date/number/currency coercion, trimming) **then** an LLM pass (existing `src/llm.py`) for rows still failing, prompted with the per-row `errors` + the target schema + allowed enum values (`paymentOrigin`, `ingestionChannel`, currencies). Anything still unresolvable → `unfixableRows`. **Open**: exact definition of "malformed", which transforms are allowed (e.g. may it invent missing `invoiceId`? — likely **no**), and whether each pass must strictly reduce the failing-row count. Recommend: each pass must reduce failures or the loop aborts early as `FAILED`.

### 4.4 Agent files
- △ `src/main.py` (new route + guard) — ⊕ `src/auth/api_key.py` (machine guard) — ⊕ `src/integrations/s3.py` — ⊕ a `fix_spend_file` function/graph — △ `pyproject.toml` (`boto3`) — △ `.env.example`.

---

## 5. helena-frontend

### 5.1 Wiring (FinOps conventions: views → components → store → service → types/i18n)
- △ `services/finops/real-spend.service.ts` — `createImportJob(file, opts)` → `POST /finops/spend/import-jobs` (returns `{ trackerId }`, unwrap `{data}`); `getTracker(trackerId)` → `GET /finops/spend/execution-trackers/:id`.
- △ `stores/finops/realSpend.ts` — `startImportJob()`, `pollTracker(trackerId)` actions (use the `wrap()` loading/error pattern).
- △ `components/finops/real-spend/SpendImportUpload.vue` — submit to the async endpoint; on success switch to a progress view.
- ⊕ `components/finops/real-spend/SpendImportProgress.vue` — polls `getTracker` every ~2 s while non-terminal; renders status, `currentStep`, `retryCount`/AGENT_FIXING banner, inserted/total, error/unfixable list; links to `SpendImportDetailView.vue` (`/finops/spend/imports/:id`) on success.
- △ `types/finops.ts` — `ExecutionTrackerStatus` enum + `ExecutionTracker` interface (money/counts as strings/numbers per existing rules).
- △ `utils/finopsConstants.ts` — status→color map.
- △ `i18n/locales/{es,pt,en}.ts` — status/step labels under `finops.enum.executionTrackerStatus.*` — **identical key structure across all three** (default `es`).
- Gating: keep `meta.financePermission`/`canFinops()`.

### 5.2 Build note
No new runtime deps expected. If any are added, **`sudo npm install`** then `npm run build` (node_modules owned by root). No test framework in this repo.

---

## 6. Cross-cutting concerns

- **Auth chain** (no user JWT exists inside a DAG run):
  - frontend → backend: existing Keycloak JWT.
  - backend → Airflow: existing `AirflowService` (JWT/token — reuse as-is; note CLAUDE.md cites `/api/v1/...dagRuns` while the synthesis saw `/api/v2/...` — **verify in `airflow.service.ts`**, but we only call the service method so the plan is unaffected).
  - DAG → backend (`/spend/batch`, tracker PATCH, webhook): `x-api-key` + optional HMAC via `AirflowApiKeyGuard`.
  - DAG → agent (`/agent/fix-spend-file`): new agent API-key guard.
- **Idempotency & all-or-nothing**: covered in §2.4 — transactional per-attempt; idempotency key blocks double-commit on network retry; fingerprint UNIQUE is the backstop.
- **Retry loop bound**: `maxAgentRetries` (default 3) carried in conf + persisted as tracker `retryCount`; terminal `FAILED` persists nothing; surface `unfixableRows`.
- **KubernetesExecutor**: all file I/O via S3; loop in one task; only small JSON via XCom.
- **Audit**: optionally emit a `FinopsAuditEvent` (existing immutable writer) at tracker terminal states.

## 7. Infra / ops checklist
- **IAM/IRSA**: the Airflow task pod SA **and** the agent pod SA need `s3:GetObject`/`s3:PutObject` on the FinOps bucket prefix `finops/imports/*`. Backend already has it.
- **Airflow connections**: confirm `aws_default`, `helena_backend`; **create `helena_agent`** (host + api-key).
- **Deploy** `dags/finops_spend_import.py` via the repo's file-sync. Do **not** enable it in the agent `WorkflowCatalog` (it's backend-internal).
- **Backend env**: `FINOPS_IMPORT_DAG_ID`, `FINOPS_IMPORT_MAX_AGENT_RETRIES`, `FINOPS_S3_BUCKET/_REGION`, `AIRFLOW_*` already present.
- **Agent env**: `AGENT_MACHINE_API_KEY`, `FINOPS_S3_BUCKET/_REGION`.

## 8. Testing
- **Backend (Jest)**: `createBatch` happy path, one-bad-row rollback (nothing persists), idempotency replay, fingerprint-dup mapping; tracker status-precedence/no-regression; `AirflowApiKeyGuard` accept/reject.
- **Agent (pytest, MemorySaver)**: API-key guard; S3 read/write mocked; normalizer coercions; LLM-repair patched; `unfixableRows` path.
- **Airflow**: DAG import check; `extract_conf` validation; loop logic unit-tested with mocked `requests`/`S3Hook` (env-var fallback path in `webhook.py` works without a live Airflow).
- **E2E (manual/staging)**: upload a deliberately malformed XLSX → watch tracker transit `PARSING→INSERTING→AGENT_FIXING→INSERTING→SUCCESS`; then an unfixable file → `FAILED`, zero rows persisted.

## 9. Phased rollout
1. **M1 — Backend data + batch insert** (no Airflow): `ExecutionTracker` model+migration, `createBatch` (txn all-or-nothing + idempotency), `/spend/batch` + `AirflowApiKeyGuard`, tracker callback + read endpoints. Unit-tested in isolation. **✅ DONE** — see "M1 status" below.
2. **M2 — Async upload + trigger**: `import-jobs` endpoint (S3 upload mandatory + `triggerDagRun`). Stub a DAG that just calls `/spend/batch` on the happy path. **✅ DONE** — see "M2 status" below.
3. **M3 — DAG full loop**: `finops_spend_import.py` with parse + tracker callbacks + agent call wired to a stub agent. **✅ DONE** — see "M3 status" below.
4. **M4 — Agent fix capability**: `boto3` + S3 integration + `/agent/fix-spend-file` + repair logic. **✅ DONE** — see "M4 status" below.
5. **M5 — Frontend**: progress component + polling + i18n; switch upload to async. **✅ DONE** — see "M5 status" below.
6. **M6 — Hardening**: HMAC, large-file chunking, audit events, retry/edge tests, staging E2E. **✅ CODE DONE** (ops steps in `OPS_REALSPEND_IMPORT.md`) — see "M6 status" below.

## 9b. M1 status (implemented 2026-06-25)
Backend-only, no Airflow/agent/frontend yet. **Typecheck clean; 70/70 Jest tests pass (12 new).**

New / changed:
- `prisma/schema.prisma` — `ExecutionTrackerStatus` enum, `ExecutionTracker` model (1:1 → `SpendImportBatch` via `import_batch_id`), `SpendBatchAttempt` idempotency ledger; reverse relation on `SpendImportBatch`.
- `prisma/migrations/20260625000000_add_execution_tracker/migration.sql` (hand-authored; dev DB applies via `db push`, prod via `migrate:deploy`).
- `src/finops/real-spend/dto/spend-batch.dto.ts` — `SpendBatchDto` / `SpendBatchRecordDto` + `SpendBatchResult` contract.
- `src/finops/real-spend/real-spend.service.ts` — `createBatch()` (idempotency replay → pre-validation → single `$transaction`, 120 s timeout, per-row failure mapping).
- `src/finops/real-spend/real-spend.controller.ts` — `POST /finops/spend/batch` (`@Public()` + `AirflowApiKeyGuard`, returns raw `SpendBatchResult`).
- `src/finops/execution-tracker/` — module, service (create / update-with-terminal-guard / findOne / findByRequestId), controller (GET human + PATCH machine), `UpdateTrackerDto`; registered in `app.module.ts`.
- `src/finops/audit/finops-audit.service.ts` — added `BATCH_CREATE` action.
- `src/config/{config.service,env.validation}.ts` — `finopsImportConfig` (`FINOPS_IMPORT_DAG_ID`, `FINOPS_IMPORT_MAX_AGENT_RETRIES`).
- Tests: `real-spend.batch.spec.ts` (7), `execution-tracker.service.spec.ts` (5).

Decisions applied: reuse `AIRFLOW_WEBHOOK_API_KEY`/HMAC (D-auth); all-or-nothing transactional insert (D4). **Deviation from §2.4**: rows without `invoiceId`/`sourceRef` are *not* hard-rejected (matches existing manual-entry behavior; the all-or-nothing rollback + idempotency key make within-run duplication impossible). Cross-upload dedup still relies on a fingerprint, so flag if a stable id should become mandatory.

Notes: repo has **no ESLint config file** → `npm run lint` is non-functional repo-wide (pre-existing, unrelated to this change). Code matches surrounding Prettier style.

## 9c. M2 status (implemented 2026-06-25)
Backend-only. **Typecheck clean; 75/75 Jest tests pass (5 new).**

Changed:
- `src/finops/real-spend/spend-import.service.ts` — new `enqueueImport(file, userId, parseOptions?)`: checksum dedup → create `SpendImportBatch` (PENDING) + `ExecutionTracker` (UPLOADING) → **mandatory** `S3Service.putObject` → `AirflowService.triggerDagRun(FINOPS_IMPORT_DAG_ID, conf)` → returns `{ trackerId, batchId, requestId, status: QUEUED }`. On S3/trigger failure, marks batch + tracker `FAILED` and throws `503`. `conf = { requestId, executionTrackerId, batchId, s3Key, fileType, maxAgentRetries, backendBaseUrl, parseOptions }`.
- `src/finops/real-spend/real-spend.controller.ts` — `POST /finops/spend/import` now calls `enqueueImport` (async; returns a tracker to poll) instead of the in-request parse+insert.
- `src/finops/real-spend/real-spend.module.ts` — imports `AirflowModule` + `ExecutionTrackerModule`; `SpendImportService` now also injects `AirflowService`, `ExecutionTrackerService`, `AppConfigService`.
- Test: `spend-import.enqueue.spec.ts` (5 — happy path, no-file, S3-unconfigured, checksum-dup, trigger-failure rollback).

Decisions applied: "replace sync with async" for the **file upload** path. Triggers Airflow **directly** (not via `AutomationService`); the DAG is backend-internal and must not be `agent-enabled`.

Scope note: the legacy synchronous `SpendImportService.import()` is **retained** only for the SAP/CLOUD external-pull path (`POST /finops/spend/ingest/:source`) — that endpoint is a different, non-file, out-of-scope flow. The Excel **file** path no longer parses/inserts in-request (parsing moves to the DAG in M3).
Resolved: **Airflow REST API is v2** (`AirflowService` calls `/api/v2/.../dagRuns`) — the code is authoritative over the older `/api/v1` note in CLAUDE.md.

Not yet runtime-verified: full Nest DI boot (no e2e/bootstrap test in the repo; module wiring reasoned to be acyclic and typechecks). The triggered DAG (`finops_spend_import`) does not exist in Airflow until **M3**, so a real upload will currently fail at the trigger step (expected at this milestone).

## 9d. M3 status (implemented 2026-06-25)
Airflow DAG in `/home/jjimenez/aiops-airflow` + a tiny backend conf addition. **6/6 pure-helper tests pass; both files byte-compile; backend still 75/75.**

New:
- `dags/finops_spend_import.py` — Airflow 3 SDK DAG (`schedule=None`, `max_active_runs=10`, tags `["finops","spend-import"]` — **not** `agent-enabled`). `extract_conf` (validates `requestId/executionTrackerId/batchId/s3Key/s3Bucket`) → `process_spend_file` (single task owning the whole loop, since KubernetesExecutor pods don't share `/tmp`) → `notify_backend` (`ALL_DONE` safety net → `/airflow-execution/webhook`).
  - Loop: PARSING → for attempt 0..maxRetries: PATCH INSERTING → `POST /finops/spend/batch` (`idempotencyKey={batchId}:{attempt}`) → `decide_outcome`: success → tracker SUCCESS & return; exhausted → tracker FAILED & raise; retry → tracker AGENT_FIXING → `POST /agent/fix-spend-file` → re-download+re-parse corrected file → attempt++.
- `dags/utils/finops.py` — `resolve_http_conn` (conn + env fallback), `_signed_headers` (x-api-key + optional HMAC matching `AirflowApiKeyGuard`), `post_spend_batch`, `patch_tracker` (best-effort), `call_agent_fix`, `download_s3_bytes` (`S3Hook`, in-memory), `read_rows` (lazy `pandas`), `normalize_rows` (alias→canonical mapping, keeps `raw`), `parse_and_normalize`, `decide_outcome`. pandas/airflow imports are lazy so the pure logic is testable without them.
- `dags/utils/finops_test.py` — 6 tests for `normalize_rows`, `_signed_headers`, `decide_outcome` (pytest- or standalone-runnable).

Backend: `spend-import.service.ts` enqueue conf now includes `s3Bucket` (`FINOPS_S3_BUCKET`) so the DAG knows the bucket; M2 test mock updated.

Connections to provision (ops, M6): `aws_default` (S3 read), `helena_backend` (exists — reused for `/spend/batch` + tracker PATCH via the same api-key/HMAC), **new `helena_agent`** (HTTP host + `AGENT_MACHINE_API_KEY`).

Parsing note: `normalize_rows` does a light alias-based header map and stashes the full original row in `raw`; rows it can't map cleanly are rejected by the backend and handed to the agent (M4, aggressive LLM mapping). Couldn't import-check the DAG locally (no Airflow in this env) — syntax validated via `py_compile`; structure mirrors the existing SDK DAGs.

## 9e. M4 status (implemented 2026-06-25)
helena-agent. **19 new tests pass; full agent suite now 36/36** (was broken at collection — see config fix).

New / changed:
- `src/auth/api_key.py` — `require_machine_api_key` FastAPI dependency (`x-api-key`, constant-time compare; 401 missing / 403 invalid / 503 unconfigured). No JWT/SSE/interrupt.
- `src/integrations/s3.py` — lazy-`boto3` `read_object`/`write_object` (IRSA, cached client).
- `src/spend_fix.py` — repair pipeline: deterministic alias map + value coercion (`_coerce_amount` handles EU/US separators, `_coerce_date` multi-format→ISO, currency, paymentOrigin synonyms→enum) **plus** an aggressive LLM column-mapping pass (`infer_column_map`, falls back to alias-only on any failure). `apply_mapping`/`find_unfixable` are pure (unit-tested). `fix_spend_file` reads S3 → maps/coerces → writes a corrected file (same format, canonical headers) → returns `{ correctedS3Key, fixedRowCount, unfixableRows }`. Aggressive defaults: `ingestionChannel→EXCEL`, unknown `paymentOrigin→OTHER`; identifying fields (amount/currency/spendDate) never fabricated. Unfixable rows are **kept** in the corrected file (honest all-or-nothing — the batch keeps failing rather than silently dropping data).
- `src/main.py` — `POST /agent/fix-spend-file` (non-SSE JSON, `Depends(require_machine_api_key)`, 404 on missing file, 500 on repair error).
- `src/config.py` — `agent_machine_api_key`, `finops_s3_bucket`, `finops_s3_region`, `spend_fix_model`; **+ `extra="ignore"`** on `Settings` (see below).
- `pyproject.toml` — added `boto3`, `pandas`, `openpyxl`. `.env.example` — new vars.
- Test: `tests/test_spend_fix.py` (19 — coercions, mapping, find_unfixable, fix_spend_file with I/O+LLM patched, api-key guard 401/403/503/ok).

Pre-existing fix: the agent `Settings` forbade unknown env vars, so the **entire** test suite was failing at collection against the dev `.env` (which carries `KEYCLOAK_CLIENT_ID`/`LANGSMITH_API_KEY`). Added `extra="ignore"` — unblocks the suite; the agent simply ignores env vars it doesn't consume.

Ops note: `uv` isn't available in this env, so **`uv.lock` was NOT regenerated** — run `uv lock` / `uv sync` (the Docker build does `uv sync`) to pull `boto3`/`pandas`/`openpyxl`. pandas/openpyxl/boto3 are lazy-imported so unit tests run without them.

## 9f. M5 status (implemented 2026-06-25)
helena-frontend (Vue 3 + Pinia). **`vue-tsc --noEmit` clean; i18n es/pt/en key parity verified.**

New / changed:
- `src/types/finops.ts` — `ExecutionTrackerStatus` enum, `TERMINAL_TRACKER_STATUSES`, `EnqueuedImport`, `ExecutionTracker` interfaces.
- `src/services/finops/real-spend.service.ts` — `import()` now returns `EnqueuedImport`; new `getTracker(id)` → `GET /finops/spend/execution-trackers/:id` (unwraps `{data}`).
- `src/stores/finops/realSpend.ts` — `importFile` returns `EnqueuedImport` (→ `lastEnqueued`); new `getTracker(id)` action + `currentTracker` ref (poll does NOT toggle the shared `loading` flag).
- `src/components/finops/real-spend/SpendImportProgress.vue` (**new**) — polls `getTracker` every 2 s while non-terminal, clears on terminal/unmount, guards overlapping polls; renders status badge, currentStep, inserted/total, agent-round/AGENT_FIXING banner, failure list, and a "view batch" link on SUCCESS; emits `done`.
- `src/components/finops/real-spend/SpendImportUpload.vue` — async submit; emits `enqueued` (EnqueuedImport); dropped the batch-counts result UI and the dead source dropdown; resets the file input after submit.
- `src/components/finops/real-spend/RealSpendPanel.vue` — renders `SpendImportProgress` (keyed by trackerId) on `enqueued`; refreshes the batch list on `done`.
- `src/utils/finopsConstants.ts` — `EXECUTION_TRACKER_STATUS_COLORS`.
- `src/i18n/locales/{es,pt,en}.ts` — `finops.enum.executionTrackerStatus.*` + `finops.realSpend.import.progress.*` (identical key structure across all three).

UX: REST polling (no `/finops` WebSocket — deferred). Lives in the existing `/helena/finops` → Real spend → Import tab, gated by `canFinops(IMPORT_SPEND)`. Note: `npm run build` (full Vite build) not run because installing/writing requires `sudo` (root-owned node_modules); typecheck via `vue-tsc` is the gate used.

## 9g. M6 status (implemented 2026-06-25)
Hardening. **Backend 76/76, agent 40/40, airflow 6/6 + py_compile — all green.**

Code:
- **Max-rows cap**: `FINOPS_IMPORT_MAX_ROWS` (default 10000). Backend `createBatch` returns `ok:false` (rowIndex -1) for oversized batches (defense-in-depth); the **DAG enforces it pre-insert** (PATCH FAILED + raise, no agent call) using `maxRows` from conf. No silent truncation.
- **Agent HMAC**: `agent_machine_hmac_secret` + `verify_hmac` (pure) in `src/auth/api_key.py`; when set, `x-hmac-signature` over the raw body is required (401 missing / 403 invalid). The DAG signs the agent call when `AGENT_MACHINE_HMAC_SECRET` is present (`resolve_http_conn(..., env_hmac=...)`). Backend endpoints already supported HMAC via `AirflowApiKeyGuard`.
- Tests: backend +1 (cap), agent +4 (verify_hmac + guard signature 401/403/ok).
- `OPS_REALSPEND_IMPORT.md` — runbook: env tables, Airflow connections (incl. **new `helena_agent`**), IRSA IAM policy JSON, migrate/lock/build commands, staging smoke test.

Operator actions (cannot be done in the dev sandbox; in the runbook): provision the `helena_agent` Airflow connection; attach IRSA S3 policy to the Airflow task + agent pods; `uv lock && uv sync` (boto3/pandas/openpyxl); `sudo npm install && npm run build`; `prisma:migrate:deploy`; staging E2E.

Deliberately NOT done (documented as follow-ups): chunked/streamed insert for files beyond the cap (kept per-row create for precise failure mapping); `/finops` WebSocket (polling instead).

## 9h. Integration-contract audit (2026-06-25)
End-to-end cross-repo verification after all milestones. Touchpoints checked and confirmed aligned: endpoint paths (`/spend/batch`, `/spend/execution-trackers/:id`, `/spend/import`), conf keys (enqueue ↔ `extract_conf`), `SpendBatchResult`/agent-response shapes, `ExecutionTrackerStatus` parity (Prisma ↔ DAG strings ↔ frontend enum ↔ i18n), and envelope wrapping (batch returns raw machine JSON; human reads wrapped `{data}`). **Two real bugs found & fixed:**

1. **(critical) Permissive batch DTO.** The global `ValidationPipe` is `whitelist + forbidNonWhitelisted + transform`. `SpendBatchRecordDto` originally enforced required `amount/currency/spendDate/paymentOrigin`, so a single malformed row would **400 the whole request** — the DAG's `raise_for_status()` would throw and the agent-fix loop would never run. Fixed: `SpendBatchRecordDto` is now all-optional strings; semantic validation moved into `createBatch` pre-validation (numeric amount, 3-char currency, valid `PaymentOrigin` enum, parseable date), returning per-row `failures`. Unknown `ingestionChannel` defaults to EXCEL. Tests +2.
2. **(robustness) Tracker stuck on DAG crash.** The terminal `notify_backend` webhook updates `AutomationRequest`/`AirflowExecutionResult`, not the `ExecutionTracker` (and this flow creates no `AutomationRequest`). If `process_spend_file` crashed before its own FAILED PATCH, the tracker would stay non-terminal and the UI would poll forever. Fixed: the DAG's `ALL_DONE` `notify_backend_task` now also PATCHes the tracker FAILED on the failure branch (idempotent — backend ignores it if already terminal).

Added a real-`ValidationPipe` integration spec (`dto/spend-batch.dto.spec.ts`, 6 tests) that proves a malformed row passes validation (→ per-row `failures`, not a 400) and that envelope-level whitelist/required checks still fire — locking in fix #1 at the HTTP layer the unit tests bypass.

Final test matrix: **backend 84/84, agent 40/40, airflow 6/6 + py_compile** — all green; backend typecheck + frontend `vue-tsc` clean.

## 9i. Live validation (2026-06-25, in dev sandbox)
Two of the riskiest pieces validated against real infra/libraries (the rest needs cluster creds):
- **Migration deploy** — `prisma migrate deploy` applied the full history (incl. `20260625000000_add_execution_tracker`) to a throwaway `postgres:16-alpine`; verified `finops_execution_trackers` (19 cols, both unique indexes, `status+created_at` index, FK→`finops_spend_import_batches` `ON DELETE SET NULL`), `finops_spend_batch_attempts`, and all 8 `ExecutionTrackerStatus` enum values in order. Container torn down.
- **Agent repair round-trip** — installed pandas/openpyxl into the agent venv and ran `fix_spend_file` for real (S3 + LLM mapping mocked) over a malformed `.xlsx`: alias mapping + coercion (`1.000,50→1000.50`, `15/02/2026→2026-02-15`, `Tarjeta→CARD`, defaults), correct `unfixableRows`/`fixedRowCount`, valid re-parseable corrected `.xlsx`. (Agent venv now has pandas/openpyxl; still pending `uv lock`.)

Still needs your env/creds: real S3 bucket, Airflow REST endpoint+JWT, the `helena_agent` connection, Keycloak — i.e. a true on-cluster end-to-end run.

## 10. Top risks / must-confirm
1. **Agent repair contract** (§4.3) — the core functional unknown; needs a written spec of "malformed" + allowed transforms before M4.
2. **All-or-nothing on large files** — single-transaction timeout; chunk inside the transaction; define a max-rows ceiling.
3. **IRSA on the agent pod** — the agent has zero S3 today; granting + verifying access is a real infra task, not just code.
4. **Airflow REST API version/auth** — verify `airflow.service.ts` actually targets the deployed Airflow's API version.
5. **Sync vs async upload coexistence** — decide whether `import-jobs` replaces or sits beside the existing synchronous `POST /finops/spend/import`.
```
