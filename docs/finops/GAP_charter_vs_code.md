# Gap Report — Charter vs. Implemented Code

> **Superseded (2026-08-07).** FinOps has left the Helena stack: the module now
> lives in `verveux-backend/src/finops/`, the spend-file repair in
> `verveux-agent/src/finops/`, and helena-backend's 19 `finops_*` tables were
> dropped by migration `20260805000000_remove_finops`. Every `helena-*` path below
> is historical. See `README.md` in this directory.

> **Comparison direction:** Charter is *behind* the code — what is already built in `helena-backend` / `helena-frontend` / `helena-agent` but is missing from, or understated in, `it-cost-center_charter.en.md`.
> **Prioritization:** Effort (to update the charter) × Value (accuracy / decision impact).
> **Generated:** 2026-07-21 · Charter version compared: 1.0.0

---

## TL;DR

The charter presents the initiative at the **Planning → Discovery** gate with "proposed scope (to confirm)", `[to confirm]` roadmap weeks, and several open *pending decisions*. The codebase shows a **substantially delivered platform**: 8 catalogs, full planned-vs-actual model, GAP/variance/projection/cash-flow endpoints, frozen-FX multi-currency, reconciliation, prorrateo with approval workflow, savings tracking, a Vue dashboard, finance RBAC, and an **async S3→Airflow→AI-agent import pipeline**. The charter's status, gate, roadmap, and "pending decisions" are stale relative to what exists.

---

## Priority 1 — High value, low effort (fix the charter now)

| # | Gap in charter | What the code already has | Where |
|---|----------------|---------------------------|-------|
| 1 | **Status/gate is stale.** Charter says gate = Planning → Discovery, scope = "proposed / to confirm", roadmap weeks all `[to confirm]`. | Full FinOps module shipped across 3 services under `/api/v3/finops/*`. | `helena-backend/src/finops/*`, migration `20260624000000_add_finops_cost_management` |
| 2 | **"7 controlled catalogs"** stated as the taxonomy (Decision 2026-07-02). | **8 catalogs** — a `ClientCatalog` is implemented in addition to the 7. | `schema.prisma:717-841` |
| 3 | **Client dimension listed as an open PENDING decision** (2026-07-02: "decide whether to create a Customer catalog"). | Already resolved in code: `ClientCatalog` + `clientId` FK + `clientClassification` enum (CLI_ESP/CLI_FAC/NA) on both `CostItem` and `RealSpendRecord`. | `schema.prisma`, catalog module |
| 4 | **Charter has no mention of governance/audit or access control.** | `FinopsAuditEvent` audit trail; `User.financeRole` + `finance-role-permissions.map.ts` RBAC gating all finance routes; frontend routes gated by `financePermission`. | `helena-backend/src/finops/audit/`, `common/permissions/finance-role-permissions.map.ts` |

---

## Priority 2 — High value, medium effort (add as scope/roadmap items)

| # | Gap in charter | What the code already has | Where |
|---|----------------|---------------------------|-------|
| 5 | **Async import pipeline + AI auto-correction is entirely absent from the charter.** This is a major built feature. | `ExecutionTracker` state machine (PENDING→UPLOADING→QUEUED→PARSING→INSERTING→AGENT_FIXING→SUCCESS/FAILED), `SpendBatchAttempt` idempotency ledger, S3 staging, and a **LangGraph agent that reads rejected rows from S3, corrects them, and writes back** — orchestrated by Airflow. | `finops/execution-tracker/`, `real-spend/spend-import.service.ts`, `finops/s3/`, `helena-agent/src/spend_fix.py`; docs `PLAN_REALSPEND_IMPORT_WORKFLOW.md`, `OPS_REALSPEND_IMPORT.md` |
| 6 | **Prorrateo described only as "spreading by percentage."** | Polymorphic `CostAllocation` (dimension = SERVICE/COUNTRY/CLIENT, sum-to-100 enforced) **plus** `DistributionFormula` with an **approval workflow** — richer + governed. | `finops/distribution-formulas/`, `schema.prisma:1057-1098` |
| 7 | **Concrete provider connectors not named** (charter says "provider APIs" generically). | Pluggable `SPEND_SOURCE_PROVIDERS`: Excel (real), **SAP** + **Cloud** HTTP connectors (config-gated), S3-mailbox (stub). Endpoints `POST /finops/spend/import` and `/spend/ingest/:source`. | `real-spend/providers/` |
| 8 | **Reconciliation described as "status (planned vs actual)."** | Full state machine: `sourceFingerprint` (@unique dedup) + `reconciliationStatus` (UNMATCHED/MATCHED/DUPLICATE/IGNORED), reconcile + unmatched endpoints. | `real-spend/reconciliation.service.ts`, `spend.util.ts` |
| 9 | **Richer categorization flags undocumented.** | `isCogs`, `isInvestmentProject`, `isTransversal`, `clientClassification`, `costNature` + DB CHECK invariants (e.g. `isCogs ⇒ serviceId`). | `schema.prisma`, `overview/overview.util.ts` |

---

## Priority 3 — Lower value / discrepancies worth a note

| # | Item | Reality in code |
|---|------|-----------------|
| 10 | Charter: "CAPEX → COGS → SG&A" as a stored classification. | **Derived at read time** (`accountTypeOf`/`costTypeOf` in `overview.util.ts`), not a persisted column — worth stating so it isn't mistaken for a stored field. |
| 11 | Charter roadmap Week 4 = dashboard. | Dashboard already exists: `FinopsView.vue` tabbed panels (Projection, CashFlow, Analysis, RealSpend, Summary) + 7 chart components + `/finops/overview` KPIs. |
| 12 | Success criteria + KPIs still `[to measure]`. | No baseline/coverage instrumentation found in code either — this gap is genuine on *both* sides; keep as pending. |

---

## Reverse gaps (code is behind charter) — for completeness

Direction requested was charter-behind-code, but three charter items are **not** implemented and should not be assumed done:

- **Infrastructure/resource inventory (item 10 of MVP scope):** *absent.* No hardware/server/cloud-resource entity — cost-center attribution lives only on cost/spend records, not on any resource inventory. **This is the largest genuine build gap.**
- **CSV/JSON file upload:** only **Excel** file parsing exists; JSON enters via HTTP batch/cloud endpoints, not file upload. No CSV parser.
- **Showback/chargeback module:** no dedicated module; only *derivable* via allocation dimensions + the overview breakdown endpoint.

---

## Recommended charter edits

1. Update §02 status and §07 roadmap to reflect a delivered/pilot state, not Planning.
2. Change "7 catalogs" → "8 catalogs (incl. Client)" in §06 and Decision log; close the Client pending decision (§08).
3. Add to §06 MVP scope: async import pipeline + AI row-correction agent; finance RBAC + audit trail; distribution-formula approval workflow.
4. Flag the three reverse gaps (resource inventory, CSV/JSON upload, showback module) explicitly in Pending/Out-of-scope so they aren't assumed complete.
