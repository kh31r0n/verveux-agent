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
    GREETING = "greeting"
    OTHER = "other"


class TriageResult(BaseModel):
    intent: IsmaelIntent
    # The question rewritten to stand on its own ("¿y qué dice Pablo?" → "¿Qué
    # dice Pablo sobre la justificación por la fe?"). Brain sees only this text,
    # never the conversation, so a follow-up must carry its own context.
    question: str = Field(default="", max_length=1000)


class SurveyAnswer(BaseModel):
    # One of the step's option keys, or "no_answer".
    answer: str = "no_answer"
    # True when the reply is a NEW theology question instead of an answer; it
    # then replaces the pending question.
    is_new_question: bool = False
