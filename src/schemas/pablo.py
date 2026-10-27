"""Contracts for the assessment agent (pablo).

Every model here is one ``generate_structured`` answer. Points are what the
model claims; code clamps them to the rubric and the backend maps the attempt
onto the activity's grade range — the model never does the arithmetic.
"""

from __future__ import annotations

import html
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _unescape(value):
    """Strings (also inside lists) without HTML entities: a plain-text Moodle
    question keeps «&iquest;», and what a model reads with entities it writes
    back with them."""
    if isinstance(value, str):
        return html.unescape(value)
    if isinstance(value, list):
        return [html.unescape(v) if isinstance(v, str) else v for v in value]
    return value


# ── Cycle composition ────────────────────────────────────────────────────────


class ComposedPart(BaseModel):
    # Must be one of the candidates the backend authorised.
    question_snapshot_id: str
    # The question adapted to an open, conversational prompt.
    prompt: str


class ComposedCycle(BaseModel):
    # One or two sentences that tie the parts together. Never a hint.
    intro: str = ""
    parts: list[ComposedPart] = Field(default_factory=list)


# ── Acknowledgement ──────────────────────────────────────────────────────────


class AnswerEcho(BaseModel):
    # A short literal fragment of the student's answer — its central idea, not
    # its best one. Code checks that it really is in the answer.
    quote: str = ""


# ── Follow-up question (one per cycle, asked to every student) ─────────────


class FollowupPlan(BaseModel):
    # A literal fragment of the answer, as in AnswerEcho.
    quote: str = ""
    # The rubric component the answer covers least; null when nothing is
    # missing (then the question deepens instead). Never points.
    target_component_id: str | None = None
    mode: str = "deepen"  # gap | deepen
    question: str = ""


# ── Scoring ──────────────────────────────────────────────────────────────────


class ComponentScore(BaseModel):
    component_id: str
    points: float
    # The rubric level/criterion the answer matched, in the rubric's words.
    criterion: str = ""
    rationale: str = ""


class QuestionVerdict(BaseModel):
    # One per question that has a reference (the teacher's "Information for
    # graders"). Code fills ``no_reference`` for the others.
    question_snapshot_id: str
    verdict: Literal["correct", "partial", "incorrect"]
    # Technical, for the teacher; never shown to the student.
    reason: str = ""


class CycleScore(BaseModel):
    # Before the components on purpose: the model judges the answer against
    # the reference first, and scores the rubric on that ground truth.
    verdicts: list[QuestionVerdict] = Field(default_factory=list)
    components: list[ComponentScore] = Field(default_factory=list)


# ── Academic feedback ────────────────────────────────────────────────────────


class OutcomeFeedback(BaseModel):
    learning_outcome_id: str
    comment: str

    _plain = field_validator("comment")(_unescape)


class QuestionFeedback(BaseModel):
    cycle_index: int
    question_snapshot_id: str
    # What was right, where the reasoning went wrong or what was missing, and
    # what to review — never the expected answer. Part and verdict are added
    # in code.
    comment: str

    _plain = field_validator("comment")(_unescape)


class AttemptFeedbackResult(BaseModel):
    overall_summary: str
    strengths: list[str] = Field(default_factory=list)
    areas_for_improvement: list[str] = Field(default_factory=list)
    per_question_feedback: list[QuestionFeedback] = Field(default_factory=list)
    per_outcome_feedback: list[OutcomeFeedback] = Field(default_factory=list)
    study_recommendations: list[str] = Field(default_factory=list)

    _plain = field_validator("overall_summary", "strengths", "areas_for_improvement", "study_recommendations")(
        _unescape
    )


# ── Class report and improvement proposals ───────────────────────────────────


class ClassReportNarrative(BaseModel):
    summary: str
    misconceptions: list[str] = Field(default_factory=list)
    teaching_recommendations: list[str] = Field(default_factory=list)
    ux_themes: list[str] = Field(default_factory=list)


class RubricClarification(BaseModel):
    rubric_version_id: str
    component_id: str
    # The criterion text as the teacher would read it in the rubric editor.
    proposed_criteria: str
    reason: str


class QuestionFlag(BaseModel):
    question_snapshot_id: str
    reason: str


class ImprovementProposals(BaseModel):
    rubric_clarifications: list[RubricClarification] = Field(default_factory=list)
    question_flags: list[QuestionFlag] = Field(default_factory=list)
    # General scoring guidance for a DRAFT of the ASSESSMENT_SCORE prompt. No
    # student answer is sent; code drops any that quotes a teacher rationale.
    prompt_guidance: list[str] = Field(default_factory=list)


# ── Background job requests (backend → agent) ────────────────────────────────


class ScoreComponent(BaseModel):
    component_id: str
    rubric_component_id: str = ""
    question_snapshot_id: str = ""
    rubric_version_id: str = ""
    part: str = ""
    name: str = ""
    max_points: float
    criteria: str = ""


class ScoreResponse(BaseModel):
    text: str | None = None
    transcript: str | None = None
    modality_used: str = "text"
    transcription_status: str | None = None


class ScoreFollowup(BaseModel):
    prompt: str
    mode: str = "deepen"
    target_component_id: str = ""
    # None = asked but never answered.
    response: ScoreResponse | None = None

    _plain = field_validator("prompt")(_unescape)


class ScoreQuestion(BaseModel):
    question_snapshot_id: str
    text: str
    # The teacher's expected answer (essay "Information for graders"); empty =
    # scored with the rubric alone. Never reaches the student.
    grader_info: str = ""

    _plain = field_validator("text", "grader_info")(_unescape)


class ScoreCycle(BaseModel):
    cycle_index: int
    learning_outcome_id: str = ""
    learning_outcome_name: str = ""
    composite_prompt: str = ""
    questions: list[ScoreQuestion] = Field(default_factory=list)
    components: list[ScoreComponent] = Field(default_factory=list)
    # None = the cycle was never reached or never answered.
    response: ScoreResponse | None = None
    # The follow-up question and its answer, scored together with the first.
    followup: ScoreFollowup | None = None

    _plain = field_validator("composite_prompt", "learning_outcome_name")(_unescape)


class CalibrationExample(BaseModel):
    rubric_version_id: str
    rubric_component_id: str
    # Anonymised by the backend before it was ever stored.
    answer: str
    points: float
    max_points: float
    rationale: str = ""


class ScoreRequest(BaseModel):
    tenant_id: str
    attempt_id: str
    assessment_id: str = ""
    # Bumped by the backend on every new scoring of the attempt (a replay of a
    # prompt draft, a rescoring after scoring_failed); the backend drops a
    # report whose version is not the one it is waiting for.
    scoring_version: int = 1
    # "publish" scores and writes feedback; "replay" only reports points, for
    # the human approving a prompt draft or rubric version (nothing published).
    mode: str = "publish"
    replay_id: str = ""
    language: str = "es"
    agent_persona_name: str = ""
    total_cycles: int = 0
    cycles: list[ScoreCycle] = Field(default_factory=list)
    calibration_examples: list[CalibrationExample] = Field(default_factory=list)
    prompts: dict[str, dict] = Field(default_factory=dict)


class InsightsDecision(BaseModel):
    rubric_version_id: str
    rubric_component_id: str
    question_snapshot_id: str = ""
    pablo_points: float
    teacher_points: float
    max_points: float
    teacher_rationale: str = ""


class InsightsRubricComponent(BaseModel):
    rubric_version_id: str
    rubric_component_id: str
    name: str = ""
    criteria: str = ""


class InsightsRequest(BaseModel):
    tenant_id: str
    assessment_id: str
    report_id: str
    language: str = "es"
    # Numbers computed by the backend (distribution, per-outcome means,
    # agreement, integrity counts, UX aggregates). Passed through as facts.
    stats: dict = Field(default_factory=dict)
    # Pablo's component rationales grouped by learning outcome name; no names.
    rationales_by_outcome: dict[str, list[str]] = Field(default_factory=dict)
    # Only sent once at least 5 surveys carry a comment (anonymity).
    ux_comments: list[str] = Field(default_factory=list)
    question_analysis: list[dict] = Field(default_factory=list)
    decisions: list[InsightsDecision] = Field(default_factory=list)
    rubrics: list[InsightsRubricComponent] = Field(default_factory=list)
    include_proposals: bool = True
    prompts: dict[str, dict] = Field(default_factory=dict)
