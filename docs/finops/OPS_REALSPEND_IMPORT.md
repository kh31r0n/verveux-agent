# Ops Runbook — RealSpendRecord Async Import Workflow

Deployment + provisioning for the upload → S3 → Airflow → agent-fix → batch-insert
pipeline. Pairs with `PLAN_REALSPEND_IMPORT_WORKFLOW.md` (design + milestone status).
Code is complete (M1–M6); the steps below are the operator actions to make it live.

## 1. Backend (helena-backend) env
| Var | Purpose | Default |
|-----|---------|---------|
| `FINOPS_S3_BUCKET` / `FINOPS_S3_REGION` | bucket for uploads/corrected files (region falls back to `AWS_REGION`) | — |
| `FINOPS_IMPORT_DAG_ID` | DAG to trigger | `finops_spend_import` |
| `FINOPS_IMPORT_MAX_AGENT_RETRIES` | agent-fix loop bound | `3` |
| `FINOPS_IMPORT_MAX_ROWS` | max rows per batch (caps the transaction) | `10000` |
| `AIRFLOW_URL` / `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD` | trigger DAGs (REST API **v2**, JWT) | — |
| `AIRFLOW_WEBHOOK_API_KEY` (≥32 chars) | guards `/spend/batch`, tracker PATCH, webhook | — |
| `AIRFLOW_WEBHOOK_HMAC_SECRET` (optional, ≥16) | enables HMAC on those endpoints | — |

Migrate: `npm run prisma:migrate:deploy` (applies `20260625000000_add_execution_tracker`). Dev uses `prisma db push`.

## 2. Agent (helena-agent) env
| Var | Purpose |
|-----|---------|
| `AGENT_MACHINE_API_KEY` | guards `POST /agent/fix-spend-file` (empty ⇒ endpoint 503s) |
| `AGENT_MACHINE_HMAC_SECRET` | optional; when set, `x-hmac-signature` over the body is required |
| `FINOPS_S3_BUCKET` / `FINOPS_S3_REGION` | same bucket the backend writes to |
| `SPEND_FIX_MODEL` | column-mapping model (default `gpt-4o-mini`) |
| `OPENAI_API_KEY` | LLM for aggressive column mapping (falls back to alias map if unset) |

**Regenerate the lock** (boto3/pandas/openpyxl were added to `pyproject.toml`):
`uv lock && uv sync` — the Docker build already runs `uv sync`, so a rebuild suffices.

## 3. Airflow connections (namespace `workflow`)
| Conn id | Type | Fields |
|---------|------|--------|
| `aws_default` | AWS | region in extra; pod IRSA role with S3 access (below) |
| `helena_backend` | HTTP | host = backend base URL; password = `AIRFLOW_WEBHOOK_API_KEY`; extra `{"hmac_secret": "..."}` (optional). **Already exists** — reused. |
| `helena_agent` | HTTP | **new** — host = agent base URL (e.g. `http://agent.workflow.svc.cluster.local:8000`); password = `AGENT_MACHINE_API_KEY`; extra `{"hmac_secret": "..."}` (optional) |

Env fallbacks (no connection): `HELENA_BACKEND_URL`, `AIRFLOW_WEBHOOK_API_KEY`, `AIRFLOW_WEBHOOK_HMAC_SECRET`, `HELENA_AGENT_URL`, `AGENT_MACHINE_API_KEY`, `AGENT_MACHINE_HMAC_SECRET`.

Deploy the DAG: sync `dags/finops_spend_import.py` + `dags/utils/finops.py` to the cluster. **Do NOT** add it to the agent `WorkflowCatalog` / tag it `agent-enabled` — it is backend-internal.

## 4. IRSA — S3 access for the Airflow task pods and the agent pod
Both need read; the agent also needs write. Scope to the imports prefix:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::<FINOPS_S3_BUCKET>/finops/imports/*" },
    { "Effect": "Allow", "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::<FINOPS_S3_BUCKET>",
      "Condition": { "StringLike": { "s3:prefix": ["finops/imports/*"] } } }
  ]
}
```
(Airflow tasks only strictly need `GetObject`; `PutObject` is the agent's. Split the policies if you prefer least-privilege per service account.)

## 5. Frontend (helena-frontend)
No new env. Build after pulling deps: `sudo npm install` then `npm run build` (node_modules is root-owned). Typecheck gate: `npx vue-tsc --noEmit`.

## 6. Smoke test (staging)
1. `/helena/finops` → **Real spend** → **Import**; upload a small, well-formed `.xlsx` (canonical headers: amount, currency, spendDate, paymentOrigin, …).
   - Tracker should transit `QUEUED → PARSING → INSERTING → SUCCESS`; "view batch" links to the import detail.
2. Upload a malformed file (e.g. headers `Monto/Moneda/Fecha/Origen`, comma decimals).
   - Expect `INSERTING → AGENT_FIXING → INSERTING → SUCCESS` (agent maps/coerces).
3. Upload an unfixable file (missing amount/date entirely).
   - Expect retries then `FAILED` with the unfixable rows listed; **zero rows persisted** (all-or-nothing).
4. Check Airflow run logs (`scripts/get_dag_logs.sh finops_spend_import`) and confirm the terminal `/airflow-execution/webhook` POST.

## 7. Known follow-ups (not blocking)
- Per-row mapping today is alias + single LLM column-map pass; revisit if files vary wildly.
- Large files beyond `FINOPS_IMPORT_MAX_ROWS` are rejected (no chunked/streamed insert yet).
- Live updates are REST polling (~2s); a `/finops` WebSocket namespace was deferred.
- `uv.lock` / full `npm run build` must be run where `uv` / `sudo` are available (couldn't be done in the dev sandbox).
