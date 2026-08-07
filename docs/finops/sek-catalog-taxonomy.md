# SEK catalog taxonomy — reference data

The concrete catalog codes SEK used, extracted from `helena-backend/prisma/seed.ts`
on 2026-08-07 before that seed's FinOps block was deleted. Preserved here because
this is the working encoding of the **SEK Cost Classification Standard v1.0**
described in `it-cost-center_charter.en.md`, and it now exists nowhere else.

> **This is one tenant's data, not a product default.** verveux-backend seeds new
> tenants from `DEFAULT_CATALOG_ENTRIES` in `src/finops/catalog/catalog-type.enum.ts`,
> which is deliberately neutral (`GLOBAL`, `GEN`, `Unassigned provider`, …). Do not
> merge the codes below into it — that would seed SEK's service lines into every
> tenant. Use them to provision a SEK tenant, via the catalog API or
> `FinopsCatalogService.provisionDefaults`.

## Dimension 1 — Servicio Principal (`service`)

| Code | Name |
|---|---|
| `SVC-MDR` | Managed Detection & Response |
| `SVC-MSS` | Managed Security Services |
| `SVC-WFX` | Workforce Extension |
| `SVC-CTEM` | Continuous Threat Exposure Mgmt. |
| `SVC-CON` | Consultoría & Awareness |
| `SVC-RED` | Red Team / Offensive Security |
| `SVC-OTR` | Servicios - Otros |
| `SVC-TRV` | Servicios - Transversal |
| `SVC-FAC` | Servicios - Factorizable |
| `CORP-FAC` | Corporativo - Factorizable por Unidad de Negocio |
| `CORP-MKT` | Corporativo - Marketing & Ventas |
| `CORP-FIN` | Corporativo - Finanzas |
| `CORP-PAC` | Corporativo - People & Culture |
| `CORP-TRV` | Corporativo - Transversal |

Service codes align to the team's COGS service reference (HU-2.2).

## Dimension 2 — País (`country`)

| Code | Name |
|---|---|
| `CL` | Chile |
| `PE` | Perú |
| `CO` | Colombia |
| `AR` | Argentina |
| `BR` | Brasil |
| `MX` | México |
| `US` | United States |
| `REG-FAC` | Regional - Factorizable |
| `REG-NC` | Regional - No Clasificable |

## Dimension 3 — Cliente / Negocio (`client`)

`code` is the Service Operation Hub ID. Real clients are provisioned from the Hub;
helena's seed carried a single example, `ACME` / "Acme Corp".

## Supporting catalogs

| Catalog | Entries |
|---|---|
| `category` | `SOFTWARE` Software · `HARDWARE` Hardware · `CLOUD` Cloud · `LICENSE` Licencia · `SERVICE` Servicio |
| `costCenter` | `TI-CORP` TI Corporativo · `SOC` SOC · `INFRA` Infraestructura |
| `businessUnit` | `INTERNAL` Interno / Transversal · `HOSTCENTER` Host Center |
| `provider` | `NEXIS` NEXIS · `AWS` Amazon Web Services · `AZURE` Microsoft Azure · `MICROSOFT` Microsoft |
| `product` | `IBM-QRADAR` IBM QRadar · `SAP` SAP · `MS365` Microsoft 365 · `EC2` AWS EC2 |

## FX baseline

Base currency came from `FINOPS_BASE_CURRENCY` (default `USD`), with two reference
rates dated `2026-01-01`, `source: 'seed'`: `USD` → `1` and `CLP` → `0.00105`.
Reference only — production needs real, date-versioned rates.
