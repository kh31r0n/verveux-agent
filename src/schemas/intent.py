
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class IntentType(str, Enum):
    SALES = "sales"
    FAQ = "faq"
    TRACKING = "tracking"
    COMPLAINT = "complaint"
    GREETING = "greeting"
    ESCALATION = "escalation"
    UNKNOWN = "unknown"

    # School-specific
    ADMISSIONS = "admissions"
    COURSE_INQUIRY = "course_inquiry"
    SCHEDULE_INQUIRY = "schedule_inquiry"

    # Camila (academic-secretary) handoff intents — any of these flips the
    # conversation to a human via POST /internal/conversations/:id/handoff.
    PAYMENT_PROOF = "payment_proof"
    CORRECTION_REQUEST = "correction_request"
    ACADEMIC_LOOKUP = "academic_lookup"
    IDENTITY_CONFLICT = "identity_conflict"

    # Restaurant-specific
    MENU_INQUIRY = "menu_inquiry"
    ORDER = "order"

    # Appointments-specific
    AVAILABILITY = "availability"
    BOOKING = "booking"
    APPOINTMENT_CANCEL = "appointment_cancel"
    APPOINTMENT_RESCHEDULE = "appointment_reschedule"

    # Leads-specific (veronica)
    LEAD_CAPTURE = "lead_capture"

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
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entities: IntentEntities = Field(default_factory=IntentEntities)
    # Additional intents detected in the same message (camila is the only
    # graph that asks the classifier to populate this). Default-empty keeps
    # checkpoints written by other graphs deserializable.
    secondary_intents: List[IntentType] = Field(default_factory=list)
    raw_text: str = ""
