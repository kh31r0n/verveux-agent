
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class IntentType(str, Enum):
    SALES = "sales"
    FAQ = "faq"
    TRACKING = "tracking"
    COMPLAINT = "complaint"
    GREETING = "greeting"
    UNKNOWN = "unknown"

class CartItemIntent(BaseModel):
    product_identifier: str
    quantity: int = 1
    notes: Optional[str] = None

class IntentEntities(BaseModel):
    items: List[CartItemIntent] = Field(default_factory=list)
    order_id: Optional[str] = None
    subject: Optional[str] = None
    description: Optional[str] = None

class StructuredIntent(BaseModel):
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    entities: IntentEntities = Field(default_factory=IntentEntities)
    raw_text: str
