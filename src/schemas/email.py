"""Contracts for the email agent (clara).

Every model the LLM returns is validated against one of these; state stores
them as ``model_dump(mode="json")`` dicts so checkpoints stay JSON-native.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Category(StrEnum):
    """Routing axis: which graph branch handles the email."""

    NOISE = "noise"
    FYI = "fyi"
    CUSTOMER_CONVERSATION = "customer_conversation"
    ACTION_REQUIRED = "action_required"


class Subcategory(StrEnum):
    """Informational only — never routes. Splits noise/fyi into buckets a human
    can filter on (Inbox Zero / Zero taxonomies)."""

    NEWSLETTER = "newsletter"
    MARKETING = "marketing"
    NOTIFICATION = "notification"
    RECEIPT = "receipt"
    OTP = "otp"
    CALENDAR = "calendar"
    COLD_EMAIL = "cold_email"
    BILLING = "billing"
    MEETING = "meeting"
    OTHER = "other"
    NONE = "none"


class Intent(StrEnum):
    QUOTATION_REQUEST = "quotation_request"
    ORDER_STATUS = "order_status"
    INVOICE_REQUEST = "invoice_request"
    COMPLAINT = "complaint"
    MEETING_REQUEST = "meeting_request"
    INFORMATION_REQUEST = "information_request"
    OTHER = "other"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sentiment(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EmailAddress(BaseModel):
    """A parsed mailbox; validation deliberately avoids a DNS/network check."""

    model_config = ConfigDict(frozen=True)

    email: str
    name: str = ""

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().lower()
        if "@" not in value or value.startswith("@") or value.endswith("@"):
            raise ValueError("address must contain a mailbox and domain")
        return value

    @property
    def domain(self) -> str:
        return self.email.rsplit("@", 1)[1]


class ParsedEmail(BaseModel):
    """The sanitized, single-message input allowed to enter the model graph."""

    gmail_message_id: str
    gmail_thread_id: str
    rfc_message_id: str | None = None
    references: list[str] = Field(default_factory=list)
    from_address: EmailAddress
    to: list[EmailAddress] = Field(default_factory=list)
    cc: list[EmailAddress] = Field(default_factory=list)
    subject: str = ""
    date: str = ""
    body: str
    attachment_names: list[str] = Field(default_factory=list)
    security_flags: list[str] = Field(default_factory=list)
    body_truncated: bool = False
    header_signals: dict[str, bool] = Field(default_factory=dict)
    gmail_labels: list[str] = Field(default_factory=list)


class SenderContext(BaseModel):
    """What the runner knows about the sender before the graph runs."""

    sender_domain: str = ""
    is_public_provider: bool = False
    is_internal: bool = False
    known_contact: bool | None = None
    contact_id: str = ""


class PrecheckResult(BaseModel):
    short_circuit: bool = False
    category: Category | None = None
    subcategory: Subcategory = Subcategory.NONE
    reasons: list[str] = Field(default_factory=list)


class TriageResult(BaseModel):
    category: Category
    subcategory: Subcategory = Subcategory.NONE
    intent: Intent
    urgency: Urgency
    sentiment: Sentiment
    summary: str = Field(min_length=1, max_length=600)
    confidence: float = Field(ge=0, le=1)
    requires_human_review: bool = False
    security_flags: list[str] = Field(default_factory=list)


class ExtractedTask(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1, max_length=4000)
    intent: Intent
    priority: Priority
    due_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    due_date_evidence: str | None = Field(default=None, max_length=500)
    requires_human: bool
    evidence_quote: str = Field(min_length=1, max_length=1000)
    evidence_valid: bool = False
    validation_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def due_date_requires_evidence(self) -> "ExtractedTask":
        if self.due_date and not self.due_date_evidence:
            raise ValueError("due_date requires due_date_evidence")
        if self.due_date_evidence and not self.due_date:
            raise ValueError("due_date_evidence requires due_date")
        return self


class ReplyDraft(BaseModel):
    body: str = Field(min_length=1, max_length=5000)
    language: str = Field(min_length=2, max_length=64)
    requires_human_review: bool = False
    safety_flags: list[str] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    """LLM-as-judge grade of one draft against natural-language criteria."""

    score: int = Field(ge=1, le=5)
    passed: bool
    reasoning: str = Field(min_length=1, max_length=1500)
