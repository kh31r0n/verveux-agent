# IT Cost Center — Full Charter

> **Charter version:** 1.0.0
> **Module:** Operations · **Status:** Active · **Current gate:** Planning → Discovery
> **Generated on:** 2026-07-02

---

## 01 · Summary

IT Cost Center is a cross-organizational FinOps initiative to monitor and govern SEK's infrastructure costs — on-prem, cloud, and independent of the consuming area.

The squad has **two distinct objectives** (visibility and optimization — standard FinOps terminology; formal framework to be defined):
1. **Visibility** — establish a unified, continuous view of infrastructure spend
2. **Optimization** — reduce infrastructure cost by 20% (the single operational target from the source document)

> **Naming note:** the name is **standardized as "IT Cost Center"** — for both the FinOps initiative and the tool that supports it (referenced internally in the data architecture as "Helena FinOps"). It refers to the **FinOps initiative** — operational management and optimization of infrastructure cost — not to the traditional IT accounting cost center.

---

## 02 · Identification and Team

### Identification

| Field | Value |
|-------|-------|
| Project | IT Cost Center |
| Product / Module | SEK Platform 2.0 / Operations |
| Current / next gate | Planning → Discovery |
| Squad status | Active |
| Delivery status | On track |
| Charter version | 1.0.0 |
| Azure DevOps link | — |

### Team

| Role | Name | Allocation | Note |
|------|------|------------|------|
| PM | Felipe Barbosa | 10% | — |
| PO | Patricio Granzotto | 50% | — |
| LE | Danilo Damilano | 20% | — |
| D | Fernanda Souza | 50% | — |
| D | Thuan | 10% | — |


### Stakeholders

| Stakeholder | Role | Responsibility |
|-------------|------|----------------|
| Felipe Barbosa | Data & Integrations Manager | Owns the datalake and the Service Platform data-architecture team. |
| Patricio Granzotto | Infrastructure Director | Owns SEK's infrastructure agenda. |
| Eduardo de Morais | Infrastructure Manager | Owns the day-to-day and operational infrastructure matters at SEK. |

---

## 03 · Problem

SEK currently lacks a unified, continuous view of the infrastructure costs consumed by its operations. On-prem infrastructure spend, cloud services (multi-provider), and resources allocated to different areas are spread across distinct tools, contracts, and cost centers, with no formal FinOps process to correlate consumption with delivered value.

Without this unified visibility, optimization decisions — rightsizing, decommissioning idle resources, and contract renegotiation — happen reactively and area-by-area, preventing SEK from capturing efficiency opportunities at scale. It also makes it impossible to produce a cash-flow view aligned with the planning Finance needs to handle provider renewals and to understand what is recurring versus extraordinary cost.

---

## 04 · Job to be Done

**Statement (when / want / outcome):** When I manage SEK's infrastructure costs (on-prem, cloud, and subscriptions), I want to categorize, project, and consolidate that spend into a single, flexible view, so I can make optimization decisions and plan cash flow proactively — instead of reactively and fragmented by area.

**Persona:** Infrastructure Management / FinOps (Patricio Granzotto, Eduardo de Morais) and Finance (for cash-flow planning and renewals).

### Problems to solve

The product must address the following:

1. **Cost categorization** — classify each expense by nature (OPEX/CAPEX), identifying the service type, cost center, and associated client.
2. **Provider spend nature and projection** — categorize whether provider spend is recurring or extraordinary, forecast renewals for cash flow, project future spend, and measure how much is actually being spent.
3. **Flexible asset ingestion** — allow new assets to be registered flexibly, whether by uploading CSV or JSON files or by connecting the tool to provider APIs to consume costs automatically.

---

## 05 · Value hypothesis

> The KPIs in this block are the source of truth for what the project promises. The same KPIs appear month over month in the Monthly Status. Without a measured baseline, mark "[to measure]" and open it as a pending decision.

| Metric | Baseline | Target | Deadline |
|--------|----------|--------|----------|
| Infrastructure cost reduction (on-prem + cloud, aggregated) | [to measure] | 20% reduction | [to define] |
| Cost-monitoring coverage (% of infra resources inventoried in the FinOps tool) | [to measure] | [to define] | [to define] |
| Areas / cost centers covered by the unified view | [to measure] | [to define] | [to define] |
| Optimization review cadence (monthly/quarterly) | [to measure] | [to define] | [to define] |

> The 20% target is the only numeric target provided by the source document. Absolute current-cost baseline and projected savings in monetary values are out of scope for this Charter (handled outside the governance document).

---

## 06 · MVP Scope

### In scope

> **Proposed scope (to confirm with the PM)** — derived from the Problem, the Job to be Done, and the Value hypothesis; promotes previously inferred items into a concrete MVP.

- **Flexible asset ingestion** — register new assets via CSV/JSON upload and via connection to provider APIs (cloud and subscriptions) for automatic cost consumption; ingest on-prem costs (DC contracts, licenses, hardware)
- **Cost categorization model** — classification by nature (OPEX/CAPEX) with a derived accounting split (CAPEX → COGS → SG&A), service type, cost center, and associated client, backed by **7 controlled catalogs** (country, service, category, cost center, business unit, provider, product) as the single cost-allocation taxonomy
- **Spend-nature classification** — flagging recurring vs. extraordinary spend by provider, renewals (auto-renew, contract term), and recording renewal dates
- **Planned vs. Actual separation + GAP/Variance** — planned obligation (CostItem) kept separate from the actual transaction (RealSpendRecord); GAP (projected − actual) and variance reports by dimension
- **Cost projection and cash flow** — renewal forecasting, projection of future spend, and measurement of actual (realized) spend, feeding Finance's cash-flow planning
- **Multi-currency with frozen FX** — normalize every amount to the base currency at record time (rate and date frozen per transaction), guaranteeing stable historical reports
- **Reconciliation and deduplication** — source fingerprint to prevent double-counting and reconciliation status (planned vs. actual)
- **Transversal cost allocation** — spreading shared cost across services by percentage (cost allocation)
- **Savings tracking** — reduction initiatives with baseline, target, and realized savings (realized = baseline − actual), evidencing progress toward the 20% target
- **Infrastructure resource inventory** (on-prem + multi-provider cloud) with attribution to cost center / consuming area
- **Visualization dashboard** — actual transactions and cost projection, aggregated by the JTBD categories (OPEX/CAPEX, provider, cost center, client, resource type)
- **Showback** — cost reporting by consuming area (chargeback is a later decision, see Out of scope)
- **Technical base on Nautilus** — module built on the Nautilus stack and specifications (SEK Platform 2.0)

---

### Out of scope

- Automatic chargeback (internal billing of areas) — the MVP delivers showback only; chargeback is a later decision (PO + sponsor + Finance)
- Headcount / hours cost management (scope of other squads, e.g., SEK Hours)

---

## 07 · Macro roadmap

> Project to be executed in **4 weeks**, already in pilot. Indicative dates — to confirm in the roadmap alignment (see Pending).

| Week | Focus | Deliverable | Status |
|------|-------|-------------|--------|
| Week 1 | Flexible asset ingestion | CSV/JSON connectors + integration with provider APIs; base inventory loaded | [to confirm] |
| Week 2 | Cost categorization | Categorization model (OPEX/CAPEX, service type, cost center, client) applied to the inventory | [to confirm] |
| Week 3 | Spend nature + projections | Recurring/extraordinary classification, renewal calendar, and spend / cash-flow projection | [to confirm] |
| Week 4 | Visualization + validation | Actual-transactions and projection-by-category dashboard; coverage validation (cloud, on-prem, subscriptions) | [to confirm] |

---

### Success criteria (binary, dated)

- [ ] [X% — to define] coverage of cloud assets
- [ ] [X% — to define] coverage of on-prem assets
- [ ] [X% — to define] coverage of subscriptions
- [ ] Visualization of actual transactions and cost projection, per the categories raised in the JTBD (OPEX/CAPEX, cost center, client, provider)
- [ ] GAP report (projected × actual) and variance available by dimension
- [ ] Savings tracking evidencing progress toward the 20% target (realized = baseline − actual)
- [ ] Module built on the Nautilus stack and specifications (SEK Platform 2.0)

---

### Pending

- Alignment of the development team with Patricio Granzotto and provisioning of hours.
- Alignment of the project's success criteria (define the X% coverage for cloud, on-prem, and subscriptions).
- Alignment of the roadmap for the next 4 weeks, given that the project is already in pilot.

---

## 08 · Decisions

### Made

| Date | Decision | Owner | Rationale |
|------|----------|-------|-----------|
| [to confirm] | Create the IT Cost Center squad as a cross-org FinOps initiative within the Operations module | [to confirm — Director + Operations leadership] | Capture the 20% infra-cost reduction opportunity and establish formal FinOps governance |
| [to confirm] | Patricio Granzotto as PO, also holding the role he already plays in SOH | [to confirm — Director + Patricio] | Leverage Operations domain knowledge; watch saturation risk |
| [to confirm] | Fernanda Souza as LE of IT Cost Center while keeping execution as Developer in SOH | [to confirm — Director + Danilo + Fernanda] | Technical continuity within the Operations family; watch bus-factor risk |
| 2026-05-19 | Initiative pillar reclassified from "SecOps Modernization & MDR" to "FinOps Initiative" in `index.md` — the squad is pure cross-org FinOps, not SecOps/MDR (record lives in `capex.yaml` per governance rule) | SP Director | Align pillar routing to the squad's real mandate |
| 2026-07-02 | Standardize the name as **"IT Cost Center"** for both initiative and tool (the internal name "Helena FinOps" from the data architecture becomes a technical reference, not the product name) | SP Director | Close the pending naming note and avoid naming friction |
| 2026-07-02 | Adopt the **7 controlled catalogs** (country, service, category, cost center, business unit, provider, product) as the single cost-allocation taxonomy | LE + PO | Resolves the pending tagging / cost-allocation-model strategy |
| 2026-07-02 | **Build on the Nautilus stack (SEK Platform 2.0)** — no external build-vs-buy evaluation | PM + LE | Resolves the pending FinOps tool/platform decision |

> See also `decisions.md` in the squad folder (if it exists).

---

### Pending

| Open since | Decision | Owner | Blocks |
|------------|----------|-------|--------|
| 2026-05-18 | **Appoint a permanent dedicated PM** (Felipe is a temporary bridge) — squad flagged as priority | SP Director | Free up Felipe (already in 7 squads); enable full operational cadence |
| 2026-05 | Define the next gate after Planning (Discovery or MVP) and respective dates | PM + PO + Director | Macro roadmap and delivery commitment |
| 2026-05 | Define the current infra-cost baseline (on-prem + cloud) and the deadline for the 20% target | PM + PO + FinOps sponsor | Value hypothesis (primary KPI) |
| 2026-05 | Define the scope of areas / cost centers covered by the MVP (all areas vs. pilot subset) | PM + PO + cross-org stakeholders | MVP scope and success criteria |
| 2026-05 | Allocate the execution team (Developers) or delivery model (own squad vs. support from the SOH squad) | Director + PO + LE | Squad execution capacity |
| 2026-05 | Define formal allocation (% and hours) for Patricio (PO) and Fernanda (LE) given the overlap with SOH | Director + PO + LE | Real capacity and saturation risk |
| 2026-05 | Appoint a **cross-org executive sponsor** (CFO/CTO/Director unbounded to Service Platform) — without a sponsor, the "area-independent" mandate is fiction | SP Director + executive | Ability to enforce decisions in areas that do not report to Service Platform |
| 2026-05 | Define the adopted **FinOps framework**: FinOps Foundation (6 principles, 3 phases Inform/Optimize/Operate) or a proprietary methodology | PM + PO + sponsor | Methodology, vocabulary, and KPIs aligned with the market |
| 2026-05 | Operationally define "infra cost": compute? storage? network? licenses? SaaS? data egress? observability? | PM + PO + LE | Inventory scope and MVP |
| 2026-05 | Define the **chargeback vs. showback** model — only report costs by area (showback) or bill areas internally (chargeback) | PO + sponsor + Finance | Governance and reduction incentives |
| 2026-05 | Reconcile the scope discrepancy: the index marks `module: Operations` and `product: SEK Platform 2.0`, but the source document describes a cross-org squad (governs costs across all of SEK) | PM + Director + sponsor | The squad's real mandate |
| 2026-07-02 | Define the **Client** dimension — the JTBD requires an "associated client", but the 7 catalogs do not include Client/Customer (Business Unit is internal); decide whether to map to an existing dimension or create a Customer catalog | PO + LE | Cost categorization by client (JTBD) |

---

## 09 · Dependencies and Risks

### Dependencies

| Dependency | Type | Who unblocks | Status |
|------------|------|--------------|--------|
| Formal PM appointment | strategic | Director | (open) |
| Access to cloud provider cost data (invoices, billing APIs) | technical | [to define — likely IT/Infra] | (open) |
| Access to on-prem cost data (DC contracts, licenses, hardware) | operational | [to define — Procurement / IT] | (open) |
| Alignment with infra-consuming areas outside Service Platform | strategic | [to define — cross-org sponsor] | (open) |
| Real capacity of Patricio (PO) and Fernanda (LE), considering SOH allocation | operational | Director + PO + LE | (open) |

---

### Structural risks

> Foundational risks. Month-level risks live in the Monthly Status. If a monthly risk recurs for 3+ months, it is probably structural and should be promoted here.

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|


---

## 10 · Operations

### Hours provisioning

> Default granularity: by profile + quarterly window. Each Quarterly Review revisits the entire grid.

| Profile | Window | Planned hours | Consumed hours | Balance | Notes |
|---------|--------|---------------|----------------|---------|-------|


---

### Communication cadence

| Ritual | Frequency | Who |
|--------|-----------|-----|

---

### Technical stack

| Component | Detail |
|-----------|--------|

---

## 11 · Estimation (capacity × demand)
