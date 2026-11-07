"""Pydantic v2 contract, schema migration (v0 -> v1), and deterministic hashing for Caleb StudyConfig v1."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

CalebModuleKey = Literal["LOCAL_MARKET", "EXPORT_EXPANSION", "CURRENT_CUSTOMERS"]
CalebStudyTier = Literal["PILOT", "STANDARD", "DEEP"]
CalebSourceKey = Literal[
    "SERPER_GOOGLE",
    "GOOGLE_PLACES",
    "PUBLIC_REGISTRY_CO_RUES",
    "PROCUREMENT_CO_SECOP",
    "PROCUREMENT_GT_GUATECOMPRAS",
    "PROCUREMENT_SV_COMPRASAL",
    "PROCUREMENT_HN_HONDUCOMPRAS",
    "PROCUREMENT_NI_SISCAE",
    "PROCUREMENT_CR_SICOP",
    "PROCUREMENT_PA_PANAMACOMPRAS",
    "PROCUREMENT_SOUTH_AMERICA_OPEN",
    "TRADE_UN_COMTRADE",
    "TRADE_WITS_TARIFF",
    "MACRO_WORLD_BANK",
    "COMPANY_WEBSITES",
    "INTERNAL_CRM_ERP",
]

CALEB_REGIONAL_COUNTRY_CODES: frozenset[str] = frozenset(
    {
        "CO",
        # Central America (7)
        "GT",
        "SV",
        "HN",
        "NI",
        "CR",
        "PA",
        "BZ",
        # South America (11)
        "EC",
        "PE",
        "CL",
        "AR",
        "BR",
        "BO",
        "PY",
        "UY",
        "VE",
        "GY",
        "SR",
    }
)

CALEB_TIER_LIMITS: dict[str, dict[str, int]] = {
    "PILOT": {
        "maxZonesOrCountries": 2,
        "maxSegments": 3,
        "maxSearchQueries": 30,
        "maxCandidateEntities": 50,
        "maxDeepScrapedEntities": 15,
        "maxTopKDeepResearch": 5,
    },
    "STANDARD": {
        "maxZonesOrCountries": 6,
        "maxSegments": 6,
        "maxSearchQueries": 120,
        "maxCandidateEntities": 250,
        "maxDeepScrapedEntities": 80,
        "maxTopKDeepResearch": 20,
    },
    "DEEP": {
        "maxZonesOrCountries": 15,
        "maxSegments": 10,
        "maxSearchQueries": 350,
        "maxCandidateEntities": 600,
        "maxDeepScrapedEntities": 200,
        "maxTopKDeepResearch": 50,
    },
}


class ProductLineModel(BaseModel):
    code: str
    name: str
    keywords: list[str] = Field(min_length=1)
    hsCodes6: list[str] | None = None
    unspscPrefixes: list[str] | None = None
    targetMarginsPct: float | None = None


class ReferenceBrandModel(BaseModel):
    name: str
    role: Literal["OWN", "DISTRIBUTED", "COMPETITOR", "MARKET_REFERENCE"]
    originCountry: str | None = None


class BusinessProfileModel(BaseModel):
    companyName: str = Field(min_length=2)
    industry: str = Field(min_length=2)
    valueProposition: str = Field(min_length=5)
    productLines: list[ProductLineModel] = Field(min_length=1)
    referenceBrands: list[ReferenceBrandModel] = Field(default_factory=list)
    disqualifiers: list[str] = Field(default_factory=list)


class SegmentDefinitionModel(BaseModel):
    code: str
    name: str
    description: str
    positiveSignals: list[str] = Field(default_factory=list)
    negativeSignals: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    priorityWeight: float = 1.0


class TaxonomyModel(BaseModel):
    segments: list[SegmentDefinitionModel] = Field(min_length=1)
    defaultSegmentCode: str


class ZoneDefinitionModel(BaseModel):
    code: str
    name: str
    country: str
    departmentOrState: str | None = None
    cities: list[str] = Field(min_length=1)
    localitiesOrNeighborhoods: list[str] | None = None
    industrialCorridors: list[str] | None = None


class TargetCountryDefinitionModel(BaseModel):
    countryCode: str
    countryName: str
    currency: str
    priorityCities: list[str] = Field(default_factory=list)
    enabledSources: list[CalebSourceKey] = Field(default_factory=list)


class GeographyModel(BaseModel):
    localZones: list[ZoneDefinitionModel] | None = None
    targetCountries: list[TargetCountryDefinitionModel] | None = None
    macroRegions: list[Literal["CO", "CENTRAL_AMERICA", "SOUTH_AMERICA"]] | None = None


class CriterionOptionModel(BaseModel):
    value: str
    points: float
    label: str


class CriterionDefinitionModel(BaseModel):
    id: str
    label: str
    weight: float
    maxPoints: float
    options: list[CriterionOptionModel] = Field(min_length=1)


class PriorityBandModel(BaseModel):
    band: Literal["A", "B", "C"]
    minScore: float
    label: str


class EntityRubricModel(BaseModel):
    maxScore: float = 100
    priorityBands: list[PriorityBandModel] = Field(min_length=1)
    criteria: list[CriterionDefinitionModel] = Field(min_length=1)


class ZoneOrCountryDimensionModel(BaseModel):
    id: str
    label: str
    weight: float
    rationalePrompt: str | None = None


class ZoneOrCountryThresholdsModel(BaseModel):
    goMin: float
    exploreMin: float


class ZoneOrCountryRubricModel(BaseModel):
    maxScore: float = 100
    thresholds: ZoneOrCountryThresholdsModel
    dimensions: list[ZoneOrCountryDimensionModel] = Field(min_length=1)


class RubricsModel(BaseModel):
    entityRubric: EntityRubricModel
    zoneOrCountryRubric: ZoneOrCountryRubricModel | None = None


class QueryTemplateModel(BaseModel):
    segmentCode: str | None = None
    template: str


class SourcesModel(BaseModel):
    enabledSources: list[CalebSourceKey] = Field(min_length=1)
    queryTemplates: list[QueryTemplateModel] | None = None


class AbcThresholdsModel(BaseModel):
    tierAPct: float
    tierBPct: float


class CustomerBaseConfigModel(BaseModel):
    sourceMode: Literal["ONE_ORDERS_AND_COMPANIES", "CSV_UPLOAD", "BOTH"]
    abcThresholds: AbcThresholdsModel
    inactivityDaysWarning: int
    inactivityDaysChurned: int
    csvUploadAssetId: str | None = None


class ExportToOneCrmModel(BaseModel):
    enabled: bool
    minPriorityBand: Literal["A", "B", "C"] = "A"
    createCompanies: bool = True
    createContacts: bool = True
    createDeals: bool = False
    pipelineId: str | None = None
    tagNames: list[str] = Field(default_factory=list)


class HumanCheckpointsModel(BaseModel):
    afterDiscovery: bool = False
    beforeCrmExport: bool = True


class DeliverablesModel(BaseModel):
    generateMarkdownReport: bool = True
    generateCsvExports: bool = True
    generateExcelWorkbook: bool = True
    exportToOneCrm: ExportToOneCrmModel
    humanCheckpoints: HumanCheckpointsModel


class LockedSafeguardsModel(BaseModel):
    allowUnverifiedContacts: Literal[False] = False
    allowHallucinatedMetrics: Literal[False] = False
    requireSourceAttribution: Literal[True] = True
    enforceBudgetHardStop: Literal[True] = True


class CalebStudyConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal[1] = 1
    studyTitle: str = Field(min_length=3, max_length=160)
    modules: list[CalebModuleKey] = Field(min_length=1)
    tier: CalebStudyTier
    businessProfile: BusinessProfileModel
    taxonomy: TaxonomyModel
    geography: GeographyModel
    rubrics: RubricsModel
    sources: SourcesModel
    customerBaseConfig: CustomerBaseConfigModel | None = None
    deliverables: DeliverablesModel
    safeguards: LockedSafeguardsModel


DEFAULT_LOCAL_MARKET_CO_CONFIG: dict[str, Any] = {
    "schemaVersion": 1,
    "studyTitle": "Estudio de Mercado Industrial B2B — Colombia",
    "modules": ["LOCAL_MARKET"],
    "tier": "STANDARD",
    "businessProfile": {
        "companyName": "Distribuidora Industrial Ejemplo S.A.S.",
        "industry": "Suministros y repuestos industriales B2B",
        "valueProposition": "Suministro mayorista de componentes mecánicos, hidráulicos y de sujeción para mantenimiento industrial y ensamblaje",
        "productLines": [
            {
                "code": "FASTENERS_AND_FIXINGS",
                "name": "Sujeción y Fijación Industrial",
                "keywords": [
                    "tornillería grado 8",
                    "espárragos B7",
                    "anclajes industriales",
                ],
                "hsCodes6": ["731815", "731816"],
                "unspscPrefixes": ["3116"],
                "targetMarginsPct": 32,
            },
            {
                "code": "POWER_TRANSMISSION",
                "name": "Transmisión de Potencia y Rodamientos",
                "keywords": ["rodamientos industriales", "chumaceras", "correas B2B"],
                "hsCodes6": ["848210", "848330"],
                "unspscPrefixes": ["3117"],
                "targetMarginsPct": 28,
            },
        ],
        "referenceBrands": [
            {"name": "MarcaPropia Pro", "role": "OWN", "originCountry": "CO"},
            {"name": "SKF", "role": "MARKET_REFERENCE", "originCountry": "SE"},
        ],
        "disqualifiers": [
            "ferretería doméstica minorista",
            "venta exclusiva al detal hogar",
        ],
    },
    "taxonomy": {
        "segments": [
            {
                "code": "WHOLESALE_IMPORTER",
                "name": "Importador / Mayorista Regional",
                "description": "Importadores y distribuidores mayoristas que abastecen subdistribuidores e industria",
                "positiveSignals": [
                    "importador directo",
                    "ventas al por mayor",
                    "cobertura nacional",
                ],
                "negativeSignals": ["venta por unidad al detal"],
                "keywords": ["importador suministros industriales", "mayorista B2B"],
                "priorityWeight": 1.0,
            },
            {
                "code": "INDUSTRIAL_MRO_CLIENT",
                "name": "Planta Industrial / Mantenimiento MRO",
                "description": "Empresas manufactureras, metalmecánicas o de ingeniería con consumo recurrente",
                "positiveSignals": [
                    "planta de producción",
                    "mantenimiento industrial",
                    "montajes electromecánicos",
                ],
                "negativeSignals": ["taller doméstico"],
                "keywords": ["metalmecánica", "mantenimiento industrial", "montajes"],
                "priorityWeight": 0.9,
            },
            {
                "code": "SPECIALIZED_DISTRIBUTOR",
                "name": "Comercializador Especializado",
                "description": "Comercializadores técnicos de nicho en zonas industriales",
                "positiveSignals": ["asesoría técnica", "catálogo industrial"],
                "negativeSignals": ["hogar y decoración"],
                "keywords": ["comercializadora industrial", "suministros técnicos"],
                "priorityWeight": 0.8,
            },
        ],
        "defaultSegmentCode": "SPECIALIZED_DISTRIBUTOR",
    },
    "geography": {
        "localZones": [
            {
                "code": "BOG_ZONA_INDUSTRIAL",
                "name": "Bogotá — Corredor Industrial Puente Aranda / Paloquemao",
                "country": "CO",
                "departmentOrState": "Cundinamarca",
                "cities": ["Bogotá"],
                "localitiesOrNeighborhoods": ["Puente Aranda", "Paloquemao", "Ricaurte"],
                "industrialCorridors": ["Calle 13", "Carrera 30"],
            },
            {
                "code": "ANT_VALLE_ABURRA",
                "name": "Antioquia — Corredor Industrial Valle de Aburrá",
                "country": "CO",
                "departmentOrState": "Antioquia",
                "cities": ["Medellín", "Itagüí", "Sabaneta"],
                "industrialCorridors": ["Autopista Sur", "Guayabal"],
            },
        ],
        "macroRegions": ["CO"],
    },
    "rubrics": {
        "entityRubric": {
            "maxScore": 100,
            "priorityBands": [
                {"band": "A", "minScore": 70, "label": "Prioridad Alta (Cuenta Clave)"},
                {"band": "B", "minScore": 45, "label": "Prioridad Media (Calificar)"},
                {"band": "C", "minScore": 0, "label": "Prioridad Baja / Monitoreo"},
            ],
            "criteria": [
                {
                    "id": "product_affinity",
                    "label": "Afinidad de Líneas de Producto",
                    "weight": 0.35,
                    "maxPoints": 35,
                    "options": [
                        {"value": "alta", "points": 35, "label": "Coincidencia directa B2B"},
                        {"value": "media", "points": 20, "label": "Líneas complementarias"},
                        {"value": "baja", "points": 5, "label": "Baja superposición"},
                    ],
                },
                {
                    "id": "operational_scale",
                    "label": "Escala Operativa y Cobertura",
                    "weight": 0.35,
                    "maxPoints": 35,
                    "options": [
                        {"value": "alta", "points": 35, "label": "Multi-sede / Importador"},
                        {"value": "media", "points": 20, "label": "Sede consolidada B2B"},
                        {"value": "baja", "points": 8, "label": "Punto único pequeño"},
                    ],
                },
                {
                    "id": "contactability_and_digital",
                    "label": "Trazabilidad Digital y Contacto Verificado",
                    "weight": 0.3,
                    "maxPoints": 30,
                    "options": [
                        {
                            "value": "alta",
                            "points": 30,
                            "label": "Web activa + teléfono/correo corporativo",
                        },
                        {
                            "value": "media",
                            "points": 15,
                            "label": "Ficha pública con teléfono verificado",
                        },
                        {"value": "baja", "points": 0, "label": "Sin canal verificable"},
                    ],
                },
            ],
        },
        "zoneOrCountryRubric": {
            "maxScore": 100,
            "thresholds": {"goMin": 70, "exploreMin": 45},
            "dimensions": [
                {"id": "demand_density", "label": "Densidad de Demanda B2B", "weight": 0.4},
                {
                    "id": "competitive_whitespace",
                    "label": "Espacio Competitivo Disponible",
                    "weight": 0.3,
                },
                {
                    "id": "logistics_accessibility",
                    "label": "Facilidad Logística y Atención",
                    "weight": 0.3,
                },
            ],
        },
    },
    "sources": {
        "enabledSources": [
            "SERPER_GOOGLE",
            "GOOGLE_PLACES",
            "PUBLIC_REGISTRY_CO_RUES",
            "PROCUREMENT_CO_SECOP",
            "COMPANY_WEBSITES",
        ],
    },
    "deliverables": {
        "generateMarkdownReport": True,
        "generateCsvExports": True,
        "generateExcelWorkbook": True,
        "exportToOneCrm": {
            "enabled": False,
            "minPriorityBand": "A",
            "createCompanies": True,
            "createContacts": True,
            "createDeals": False,
            "tagNames": ["caleb-estudio-local"],
        },
        "humanCheckpoints": {
            "afterDiscovery": False,
            "beforeCrmExport": True,
        },
    },
    "safeguards": {
        "allowUnverifiedContacts": False,
        "allowHallucinatedMetrics": False,
        "requireSourceAttribution": True,
        "enforceBudgetHardStop": True,
    },
}


def migrate_study_config_to_v1(raw: Any) -> dict[str, Any]:
    """Upgrades legacy/unversioned study config dicts (v0/None) to v1."""
    if not isinstance(raw, dict):
        return {}
    migrated = copy.deepcopy(raw)
    if migrated.get("schemaVersion") in (None, 0):
        migrated["schemaVersion"] = 1
    if not isinstance(migrated.get("safeguards"), dict):
        migrated["safeguards"] = {
            "allowUnverifiedContacts": False,
            "allowHallucinatedMetrics": False,
            "requireSourceAttribution": True,
            "enforceBudgetHardStop": True,
        }
    return migrated


def validate_study_config_v1(
    raw: Any,
) -> tuple[bool, CalebStudyConfigV1 | None, list[dict[str, str]]]:
    """Validates a raw StudyConfig dict and returns (valid, model, field_errors)."""
    if not isinstance(raw, dict):
        return (
            False,
            None,
            [
                {
                    "field": "root",
                    "code": "INVALID_TYPE",
                    "message": "La configuración del estudio debe ser un objeto JSON.",
                }
            ],
        )

    errors: list[dict[str, str]] = []

    raw_safeguards = raw.get("safeguards")
    if isinstance(raw_safeguards, dict):
        if (
            raw_safeguards.get("allowUnverifiedContacts") is not False
            or raw_safeguards.get("allowHallucinatedMetrics") is not False
            or raw_safeguards.get("requireSourceAttribution") is not True
            or raw_safeguards.get("enforceBudgetHardStop") is not True
        ):
            errors.append(
                {
                    "field": "safeguards",
                    "code": "LOCKED_SAFEGUARD_VIOLATION",
                    "message": "Las salvaguardas de integridad están bloqueadas y no pueden desactivarse.",
                }
            )

    migrated = migrate_study_config_to_v1(raw)
    model: CalebStudyConfigV1 | None = None
    try:
        model = CalebStudyConfigV1.model_validate(migrated)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ())) or "root"
            errors.append(
                {
                    "field": loc,
                    "code": str(err.get("type", "VALIDATION_ERROR")).upper(),
                    "message": err.get("msg", "Campo inválido"),
                }
            )

    if model is not None:
        limits = CALEB_TIER_LIMITS[model.tier]
        if len(model.taxonomy.segments) > limits["maxSegments"]:
            errors.append(
                {
                    "field": "taxonomy.segments",
                    "code": "TIER_LIMIT_EXCEEDED",
                    "message": f"El tier {model.tier} permite máximo {limits['maxSegments']} segmentos.",
                }
            )
        seg_codes = {s.code for s in model.taxonomy.segments}
        if model.taxonomy.defaultSegmentCode not in seg_codes:
            errors.append(
                {
                    "field": "taxonomy.defaultSegmentCode",
                    "code": "INVALID_DEFAULT_SEGMENT",
                    "message": "defaultSegmentCode debe pertenecer a taxonomy.segments.",
                }
            )

        local_zones = model.geography.localZones or []
        target_countries = model.geography.targetCountries or []
        if "LOCAL_MARKET" in model.modules and len(local_zones) == 0:
            errors.append(
                {
                    "field": "geography.localZones",
                    "code": "REQUIRED_FOR_LOCAL_MARKET",
                    "message": "LOCAL_MARKET requiere al menos una zona en geography.localZones.",
                }
            )
        if "EXPORT_EXPANSION" in model.modules and len(target_countries) == 0:
            errors.append(
                {
                    "field": "geography.targetCountries",
                    "code": "REQUIRED_FOR_EXPORT_EXPANSION",
                    "message": "EXPORT_EXPANSION requiere al menos un país en geography.targetCountries.",
                }
            )
        for tc in target_countries:
            if tc.countryCode not in CALEB_REGIONAL_COUNTRY_CODES:
                errors.append(
                    {
                        "field": "geography.targetCountries",
                        "code": "UNSUPPORTED_COUNTRY_CODE",
                        "message": f"País no soportado: {tc.countryCode}",
                    }
                )
        if len(local_zones) + len(target_countries) > limits["maxZonesOrCountries"]:
            errors.append(
                {
                    "field": "geography",
                    "code": "TIER_LIMIT_EXCEEDED",
                    "message": f"El tier {model.tier} permite máximo {limits['maxZonesOrCountries']} zonas/países.",
                }
            )
        if "CURRENT_CUSTOMERS" in model.modules and model.customerBaseConfig is None:
            errors.append(
                {
                    "field": "customerBaseConfig",
                    "code": "REQUIRED_FOR_CURRENT_CUSTOMERS",
                    "message": "CURRENT_CUSTOMERS requiere customerBaseConfig.",
                }
            )
        weight_sum = sum(c.weight for c in model.rubrics.entityRubric.criteria)
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(
                {
                    "field": "rubrics.entityRubric.criteria",
                    "code": "WEIGHT_SUM_MISMATCH",
                    "message": f"Los pesos de la rúbrica deben sumar 1.0 (actual: {weight_sum:.2f}).",
                }
            )

    if errors:
        return False, None, errors
    return True, model, []


def _canonicalize_json(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float, str)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_canonicalize_json(v) for v in value) + "]"
    if isinstance(value, dict):
        items = [
            f"{json.dumps(k, ensure_ascii=False)}:{_canonicalize_json(value[k])}"
            for k in sorted(value.keys())
            if value[k] is not None
        ]
        return "{" + ",".join(items) + "}"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compute_study_config_hash(config: dict[str, Any] | CalebStudyConfigV1) -> str:
    payload = (
        config.model_dump(exclude_none=True)
        if isinstance(config, CalebStudyConfigV1)
        else config
    )
    canonical = _canonicalize_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
