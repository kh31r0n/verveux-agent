"""Contracts for the theology agent (ismael).

The survey answers are statistics: the option keys below are what lands in
``Contact.profileData.ismael`` and what any report groups by, so they are fixed
here rather than taken from tenant-editable prompts. ``no_answer`` records a
refusal or an unreadable reply — the survey never asks twice.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class IsmaelIntent(StrEnum):
    THEOLOGY = "theology"
    # How to use the Moodle classroom, a technical problem, or an
    # administrative errand (enrolment, payments) to redirect.
    MOODLE_SUPPORT = "moodle_support"
    # Wants to talk to / write to their teacher.
    CONTACT_TEACHER = "contact_teacher"
    GREETING = "greeting"
    OTHER = "other"


class TeacherTopic(StrEnum):
    """What a "contact my teacher" request is about, so ismael can offer to
    help first with the branch that fits."""

    THEOLOGY = "theology"
    MOODLE_SUPPORT = "moodle_support"
    NONE = "none"


class TriageResult(BaseModel):
    intent: IsmaelIntent
    # The question rewritten to stand on its own ("¿y qué dice Pablo?" → "¿Qué
    # dice Pablo sobre la justificación por la fe?"). Brain sees only this text,
    # never the conversation, so a follow-up must carry its own context. For
    # moodle_support it is the request in one sentence; for contact_teacher,
    # the reason or the message for the teacher, when the user gave one.
    question: str = Field(default="", max_length=1000)
    teacher_topic: TeacherTopic = TeacherTopic.NONE
    # Id of the institution FAQ (among the candidates shown to triage) that
    # answers the message — or the teacher request's reason — or "" when none
    # does. Validated against the candidates in code; never trusted blindly.
    faq_id: str = Field(default="", max_length=64)


class MoodleSupportOutcome(StrEnum):
    ANSWERED = "answered"
    NEEDS_TICKET = "needs_ticket"
    REDIRECT = "redirect"


class MoodleSupportResult(BaseModel):
    reply: str = Field(max_length=2000)
    outcome: MoodleSupportOutcome = MoodleSupportOutcome.ANSWERED
    # First-person description for the institution's support form; only
    # meaningful with needs_ticket. The form link and the user's data are
    # appended in code, never by the model.
    ticket_description: str = Field(default="", max_length=1000)


class TeacherStepAnswer(BaseModel):
    # 1-based number of the option the user chose; 0 when none fits.
    choice: int = 0
    # True when the reply is something else entirely (a new question), which
    # abandons the teacher flow.
    is_new_question: bool = False


class TeacherDraft(BaseModel):
    # The message for the teacher, in the student's voice.
    message: str = Field(max_length=1500)


class SurveyAnswer(BaseModel):
    # One of the step's option keys, or "no_answer".
    answer: str = "no_answer"
    # True when the reply is a NEW theology question instead of an answer; it
    # then replaces the pending question.
    is_new_question: bool = False


class FaqAnswer(BaseModel):
    # The FAQ's answer adapted to the exact question, adding nothing it lacks.
    reply: str = Field(max_length=2000)
