# Pablo: AI-Powered Assessment Agent for Moodle

## 0 — Revision Notes (2026-10-25)

### 0.1 — New requirements and how this plan meets them

| # | Requirement | Where | How |
|---|---|---|---|
| R1 | The teacher's review decision is made entirely within Moodle, through the plugin. | §9, §13 | Review queue, review page and decision form are Moodle pages of the activity. The plugin writes the gradebook in the same request. The CRM has no review UI. |
| R2 | The UX survey comes before scoring. | §3, §8.2 | Order is now cycles → survey → scoring → feedback. The survey is a fixed form, still skippable, shown before the student has seen any score. |
| R3 | With no Integrity Alert, the agent grades on the activity's configurable min/max. | §4.3, §6.3 | The activity's grade range (`grade_min`…`grade_max`, e.g. 1.0–5.0) is set in Moodle. Pablo scores rubric components and code maps the result onto that range. An alert does **not** hold the grade: it is published and the attempt is flagged into the teacher's review queue. |
| R4 | The timer pauses while the agent responds; the student chooses when to continue, then gets the next question. | §3.1 | The backend owns the clock and counts only the time a question is on screen and open for answering. After each answer the attempt waits on a **Continue** button. The next question is composed only after the click and the timer restarts when it is shown. |
| R5 | During review the teacher sees the full conversation in Moodle. | §9.2 | The review page renders the attempt's whole transcript (prompts, answers, audio transcripts, Pablo's messages, pause/continue times) from the backend's message store, server-side, inside Moodle. |
| R6 | Global feedback for the teacher once every student has completed the activity. | §10 | Class report, triggered when every student who can attempt has a finished attempt (with a close-date fallback and a "Generate now" button). Shown in Moodle. |
| R7 | The agent improves itself, under human supervision. | §11 | Only teacher decisions feed improvements (never Pablo's own unreviewed output). Proposals take three forms: calibration examples, rubric clarifications and prompt drafts. None applies until a human approves it, every grade records the versions it was produced with, and nothing is ever regraded retroactively. |

### 0.2 — Decisions confirmed with the product owner

- **Min/max = grade scale** (Moodle's `grademin`/`grademax`), not an auto-publish band and not a question count.
- **On an alert: publish and flag.** The grade is published as usual; the alert opens a review case in Moodle.
- **UX survey: shown first, skippable.** It precedes scoring, but skipping stays allowed.

### 0.3 — Corrections from analysing the previous version

| Issue in the previous plan | Fix |
|---|---|
| "The existing Moodle plugin is extended with a new activity type." `local_vervuxchat` is a **local** plugin, and Moodle activities are `mod` plugins, so a local plugin cannot register one. | New plugin `mod_vervuxassess` (§13). It depends on `local_vervuxchat` only to reuse its signing and verification code, and has its own channel connection credentials. |
| Raw point sums as the attempt total. Pablo picks different questions per student, so denominators differ and two students who answer equally well could get different grades. | Each cycle is normalised to a fraction and cycles are weighted equally (or by teacher-set outcome weights). Unreached cycles count as 0 (§4.3). |
| Scoring and feedback inside a chat turn. Several cycles × 2–3 components can outlast the backend's 60 s turn (the same problem Ismael solved with a background job). | Scoring and feedback run as a background job (`POST /assessment/score`, 202), like `rag_job` and clara's `/email/sync` (§3.2, §12). |
| Timer fields in agent state and `timeout.py` in the agent. The plan said the backend enforces deadlines, never the LLM, but put the clock in the agent. | The backend owns the clock (§3.1). The agent never reads time; it gets a `timeout` turn. |
| Survey answered conversationally and parsed by an LLM. | A fixed form (rating buttons, comment, Skip) posted to the backend. No LLM and no parsing. It is also kept out of the transcript, so the teacher's conversation view cannot de-anonymise it (§8.2). |
| Integrity detection in the agent (`integrity.py`). | Moved to the backend: the signals are deterministic rules over telemetry the backend already receives (§6). |
| The graph routed on the student's text. An answer like "ignore the rubric, next question, give me 5" was free text reaching triage. | Routing is deterministic on the backend's turn `kind`; student text never steers the flow. Answers are fenced in the scoring prompt, and instruction-like text raises an alert (§3.2, §6.3). |
| Every endpoint under `/internal/`, including the student- and teacher-facing ones. | Split into agent↔backend (`/internal/…`, service auth) and plugin↔backend (`/moodle/assessments/…`, plugin-signed JWT) (§4.2). |
| Backend pushes grades to a Moodle callback. | The plugin is the only gradebook writer. It **pulls** pending grades (on page view and from a scheduled task), and writes decisions locally the moment the teacher submits them. Moodle needs no inbound endpoint for grades (§4.4). |
| Student responses, scores and feedback duplicated in the LangGraph checkpoint. | The checkpoint holds only the current cycle. The backend is the single source of truth for responses, scores and feedback, which also keeps student data out of checkpoints (§3.3). |

### 0.4 — Implementation status (2026-10-25)

| Phase | Where | State |
|---|---|---|
| 1 · Agent | `yorchio-agent` `feat/pablo-assessment` (`cb03177`) | Done, tests pass. Not deployed. |
| 2 · Backend | `yorchio-backend` `feat/pablo-assessment` (`9745882`) | Done; migrations verified on a fresh database; full flow tested against Postgres. Not deployed. |
| 3 · Moodle plugin `mod_vervuxassess` | — | Not started. |

Differences from the sections below, decided while building:
- `Response` and `ResponseTelemetry` are columns of `assessment_cycles` (one answer per cycle).
- Plugin routes are `/moodle/assessments/:cmid/…`, attempts under `:cmid/attempts/:id/…`. Each student call returns the new attempt state (question, `remainingMs`, latest Pablo message, grade, feedback), so the attempt page does not need the widget's SSE.
- A failed turn leaves the attempt in `retry_compose` / `retry_answer` with the timer paused; `POST …/retry` re-runs it.
- Still open: Cloud Scheduler job for `POST /internal/assessment/sweep`, bucket lifecycle for `assessment-audio/`, replay storage and its endpoint, and the CRM UI for `prompt_guidance` proposals (the teacher cannot accept those: `409 ADMIN_DECIDES`).

---

## 1 — Summary

Create **Pablo** as a permanent, isolated agent (`pablo`) with a new `ASSESSMENT` agent type, delivered as a **dedicated Moodle activity** (`mod_vervuxassess`). It is not a quiz question type and not an extension of the chat bubble. The teacher configures everything in Moodle: question import, learning-outcome mapping, rubrics, the grade range (min/max), timing, attempts and accommodations. The student opens the activity and completes a series of cycles (text, audio, or both). Each answer pauses the timer, and the student presses **Continue** for the next question. After the last cycle the student fills in a short, skippable UX survey. Pablo then scores the attempt, the grade is published on the activity's scale, and Pablo delivers academic feedback. Attempts with integrity alerts, student review requests or spot-check samples land in the teacher's **review queue in Moodle**, where the teacher reads the full conversation and decides. When every student has finished, the teacher gets a **class report**. Teacher decisions feed a **human-approved improvement loop**.

Pablo is **strictly conversational on the student side**: the chat graph only talks to students during an active attempt. Teacher-facing outputs (class report, improvement proposals) are background jobs of the same service. They never chat, and they are shown in Moodle.

Each tenant can run Pablo alongside Ismael (or any other agent) on **separate channel connections**; they never share state.

> [!IMPORTANT]
> **V1 scope is full.** Audio, integrity monitoring, teacher review in Moodle, accommodations, the class report and the supervised improvement loop are all in-scope for the first release.

---

## 2 — Architecture Overview

```mermaid
flowchart LR
  subgraph Moodle["Moodle"]
    Activity["mod_vervuxassess<br/>attempt UI · review · reports"]
    QBank["Question Bank"]
    Gradebook["Gradebook"]
    Task["Scheduled task<br/>grade pull · completion check"]
  end

  subgraph Backend["NestJS Backend"]
    MoodleAPI["/moodle/assessments/* (plugin-signed)"]
    Domain["Assessment domain<br/>clock · alerts · grade mapping"]
    Msgs["Conversations / Messages<br/>(one conversation per attempt)"]
    Prompts["ai_prompts versions<br/>DRAFT / ACTIVE / ARCHIVED"]
  end

  subgraph Agent["Python Agent Service"]
    Chat["pablo chat graph<br/>/chat/stream"]
    Score["scoring + feedback job<br/>/assessment/score"]
    Insights["class report + proposals job<br/>/assessment/insights"]
    STT["Google Cloud STT v2"]
  end

  QBank -->|"versioned import"| Activity
  Activity -->|"config, answers, survey, decisions"| MoodleAPI
  MoodleAPI --> Domain
  Domain -->|"turns: start / answer / continue / timeout"| Chat
  Chat --> STT
  Chat -->|"cycles, responses"| Domain
  Domain -->|"after survey"| Score
  Score -->|"component points, feedback"| Domain
  Domain -->|"all students done"| Insights
  Insights -->|"report, proposals"| Domain
  Domain --- Msgs
  Prompts -.->|"active version"| Chat
  Task -->|"pull pending grades"| MoodleAPI
  Task --> Gradebook
  Activity -->|"teacher decision → grade_update()"| Gradebook
```

---

## 3 — Attempt Flow & Threading Model

| Concept | Mapping |
|---|---|
| **Attempt** | One backend conversation and one LangGraph thread. The backend opens a **new conversation per attempt**, not the contact's chat conversation, so the thread, the transcript the teacher reads, and the billing all stay per attempt. |
| **Cycle** | Pablo presents a composite prompt; the student answers (text/audio/both); Pablo acknowledges; the attempt waits for **Continue**. |
| **UX survey** | Fixed form after the last cycle, **before scoring**, skippable. |
| **Scoring** | Background job after the survey is submitted or skipped. Pablo scores components; the backend maps the result to the activity's grade range and publishes it. |
| **Academic feedback** | Same background job. It is posted into the attempt's conversation and stored for later. |

The flow is **strictly sequential**: cycle 1 → answer → pause → Continue → cycle 2 → … → last answer → UX survey → scoring → grade published (flagged if an alert exists) → academic feedback. The student cannot navigate between cycles or go back.

```mermaid
sequenceDiagram
    participant S as Student (Moodle UI)
    participant B as Backend
    participant P as Pablo (chat graph)
    participant J as Pablo (scoring job)

    S->>B: Start attempt
    B->>P: turn kind=start
    P-->>B: composite prompt (cycle 1)
    B-->>S: question shown — timer RUNS
    S->>B: Answer (+ telemetry)
    Note over B: timer PAUSED on receipt
    B->>P: turn kind=answer
    P-->>B: acknowledgement (+ transcript if audio)
    B-->>S: "Continue" button — timer still paused
    S->>B: Continue (when the student chooses)
    B->>P: turn kind=continue
    P-->>B: composite prompt (cycle 2)
    B-->>S: question shown — timer RUNS
    Note over S,B: … repeat to the last cycle …
    B-->>S: UX survey form (no timer)
    S->>B: Survey answered or skipped
    B->>J: POST /assessment/score (202)
    J-->>B: component points + feedback
    Note over B: map to [grade_min, grade_max], open review case if alerts
    B-->>S: grade + academic feedback
```

### 3.1 — Timer & Pause (backend-owned)

The backend is the only clock; the client only displays `remaining_ms` from the server and auto-submits at zero.

| Phase | Timer | Ends when |
|---|---|---|
| `answering` | **Running** | The answer is received (server time = `submitted_at`) or the cycle times out. |
| `agent_working` | Paused | Pablo's acknowledgement is stored (transcription, if any, happens here). |
| `awaiting_continue` | Paused | The student presses **Continue** (`continue_requested_at`). No cap other than the activity close date. |
| `composing` | Paused | The next question is stored and dispatched (`presented_at`). The timer restarts here, so composition latency never costs the student time. |
| `survey` | None | Survey submitted or skipped (or abandoned, §8.2). |

- **Time used** = Σ (`submitted_at` − `presented_at`) over cycles. It is derived from three timestamps on `Cycle`, so no event table is needed.
- **Deadlines**: cycle deadline = `presented_at` + min(cycle limit, attempt time remaining), both × the accommodation's `extra_time_multiplier`. Submissions are accepted up to the deadline + 5 s of grace for network latency.
- **Cycle timeout**: the client auto-submits the draft at zero (marked `timed_out`). If the client has vanished, a backend sweeper closes the cycle unanswered. Either way the attempt moves to `awaiting_continue` with the notice "time is up for this question".
- **Attempt timeout**: the remaining cycles are recorded as unreached and the attempt goes straight to the survey.
- **Leaving the page** does not pause anything: only a received answer does. Otherwise closing the tab would be an unlimited pause *with the question visible*.
- **The pause cannot be exploited**: the next question does not exist on the client (it is not composed yet) until Continue is pressed, and the previous answer is already locked.
- **Activity close date** is a hard stop regardless of phase: open cycles are closed and the attempt goes to the survey or, if the student is gone, is treated as an abandoned survey (§8.2).
- Telemetry (§6.2) only counts during `answering`. Focus loss while paused is not a signal.

### 3.2 — Graph Topology (chat graph)

```
START → pablo_triage                (deterministic, on the backend's turn kind — never on student text)
          ├── kind=start | continue → pablo_next_cycle     → END
          ├── kind=answer           → pablo_receive_answer → END
          ├── kind=timeout          → pablo_timeout        → END
          └── unexpected / failure  → pablo_error          → END
```

- `pablo_next_cycle`: fetches authorised candidates for the next learning outcome, selects 2–3 and composes the prompt (one `generate_structured`). It reports the cycle to the backend.
- `pablo_receive_answer`: transcribes audio if present, reports the response, and replies with a fixed acknowledgement. For audio it adds the transcript, so the student sees what will be evaluated. On the last cycle it adds "that was the last question" and the backend shows the survey. **No scoring and no evaluative comment** between cycles.
- `pablo_timeout`: writes the fixed timeout notice; no LLM.
- Scoring, feedback and the class report are **not** graph nodes. They are background jobs (§12.2), so a long attempt never hits the backend's 60 s turn limit.

### 3.3 — `state["pablo"]` Schema

`chat_stream` resets `domain_state` on every request, so it carries only this turn's `assessment_turn` (which can therefore never be replayed from the checkpoint). What must persist lives in a top-level `pablo` dict, the way Ismael keeps `ismael`. The checkpoint keeps only what the next turn needs; responses, scores and feedback live in the backend.

```python
{
    "assessment_id": str,
    "attempt_id": str,
    "total_cycles": int,
    "current_cycle": int,              # 0-indexed
    "modality": str,                   # "text" | "audio" | "both" (after accommodation override)
    "phase": str,                      # "answering" | "awaiting_continue" | "finished" (mirror; backend is authoritative)
    "used_question_ids": list[str],    # never re-ask within the attempt (backend also enforces)
    "cycle": {                         # current cycle only
        "learning_outcome_id": str,
        "composite_prompt": str,
        "component_map": list,         # [{component_id, question_snapshot_id, rubric_version_id}]
        "prompt_provenance": dict,     # {prompt_key, prompt_id, prompt_version, prompt_sha, model_id}
    } | None,
}
```

---

## 4 — Backend Domain (NestJS)

### 4.1 — Data Model

All additive; no modifications to existing tables except a nullable `attemptId` link on `Conversation`.

| Entity | Purpose |
|---|---|
| `Assessment` | Activity config synced from Moodle: availability window, cycle/attempt duration, max attempts, `grade_method` (highest/last), **`grade_min`, `grade_max`, `grade_decimals`**, optional `grade_pass`, review period, modality, `spot_check_rate`, Moodle `cmid`/course id. The grade range is locked once the first attempt starts. |
| `AssessmentLearningOutcome` | Learning outcomes, with optional weight (default equal). |
| `ImportedQuestion` | Immutable versioned snapshot of a Moodle question, mapped to outcome, difficulty, topic. |
| `Rubric` / `RubricVersion` | Teacher-authored rubric. Each edit (including an accepted improvement proposal) creates a new version; evaluations reference the version used. |
| `Attempt` | Student attempt: status (`active` / `survey` / `scoring` / `scored` / `abandoned`), attempt number, conversation id, thread id, time used. |
| `Cycle` | Selected snapshot ids, composite prompt, component map, **`presented_at`, `submitted_at`, `continue_requested_at`**, `timed_out`, prompt provenance. |
| `Response` | Text, transcript, modality used, transcription status. |
| `ResponseTelemetry` | Aggregate metrics (§6.2). |
| `ComponentEvaluation` | Points, max, criterion, rationale, model + prompt + rubric version + calibration-set version. |
| `AttemptGrade` | Cycle fractions, attempt fraction, **grade on the activity scale**, source (`pablo` / `teacher_decision`), version (for idempotent sync), `synced_at`. |
| `IntegrityAlert` | Signal, severity, cycle, details. Never changes a grade. |
| `ReviewCase` | One review case, opened by `integrity_alert`, `student_request` or `spot_check`. Sources on the same attempt merge into one open case. |
| `ReviewDecision` | Immutable: confirm / change grade, new grade, rationale, feedback to the student, Moodle teacher user id, `use_as_example` flag. |
| `StudentAccommodation` | Extra time multiplier, modality override, disabled signals. |
| `AttemptFeedback` | Academic feedback + provenance. |
| `UxSurveyResponse` | Rating (nullable), comment (nullable), `skipped`, `abandoned`. Linked to the attempt for idempotency only; never exposed per attempt. |
| `ClassReport` | Versioned class-level report (§10), with an `outdated` flag. |
| `CalibrationExample` | Approved, anonymised answer + teacher grade + rationale for one rubric version (§11). |
| `ImprovementProposal` | Kind, payload, evidence (decision ids), status (`proposed` / `accepted` / `edited` / `rejected` / `reverted`), approver, replay result. |

### 4.2 — API Contracts

The **backend enforces** enrolment, permissions, deadlines, attempt limits, the grade range and persistence, never the LLM.

**Backend → agent** (service auth, `verify_service_caller`):

| Endpoint | Purpose |
|---|---|
| `POST /chat/stream` | Attempt turns. The body carries `assessment_turn: {kind, attempt_id, cycle_index, answer?}`. |
| `POST /assessment/score` | 202. Score + feedback for one attempt. Idempotent on `attempt_id` + `scoring_version`. Payload: responses, rubric versions, calibration examples, grade range. Also used in `replay` mode (no publication) for §11. |
| `POST /assessment/insights` | 202. Class report + improvement proposals for one assessment. Idempotent on `report_id`. |

**Agent → backend** (`/internal/…`, service auth):

| Endpoint | Method | Purpose |
|---|---|---|
| `/internal/assessment/attempts/:id/next-cycle` | POST | Authorised candidate questions for the next outcome (backend selects eligible ones, Pablo picks 2–3). |
| `/internal/assessment/attempts/:id/cycles/:n` | POST | Report the composed cycle (prompt, component map, provenance). |
| `/internal/assessment/attempts/:id/cycles/:n/response` | POST | Report the transcript/text as evaluated. |
| `/internal/assessment/attempts/:id/score` | POST | Component points + rationale. The backend clamps, maps to the scale, publishes, and evaluates score-based alerts. |
| `/internal/assessment/attempts/:id/feedback` | POST | Academic feedback (also posted as a message via the existing `agent-messages` endpoint, idempotent on `jobId`). |
| `/internal/assessment/assessments/:id/insights` | POST | Class report + proposals. |

**Moodle plugin → backend** (`/moodle/assessments/…`). Each call carries a plugin-signed HS256 JWT, the same scheme as the MOODLE channel's identity token, with its own audience `vervux:moodle:assessment` and claims `sub` (Moodle user id), `cmid` and `cap` (`attempt` or `review`). The plugin signs `cap` only after its own `has_capability()` check. The backend never trusts a role sent from the browser.

| Endpoint | Caller | Purpose |
|---|---|---|
| `PUT /moodle/assessments/:cmid` | teacher (server-side) | Sync activity settings, outcomes, question snapshots, rubrics, accommodations. |
| `POST /moodle/assessments/:cmid/attempts` | student | Start or resume. Returns the attempt state and a stream token (widget SSE transport). |
| `POST /moodle/attempts/:id/answer` | student | Text / audio reference + telemetry. Pauses the timer. |
| `POST /moodle/attempts/:id/continue` | student | Ask for the next question. |
| `GET /moodle/attempts/:id/state` | student | Phase, `remaining_ms`, current prompt (resume after a crash). |
| `POST /moodle/attempts/:id/survey` | student | Rating + comment, or skip. Idempotent; triggers scoring. |
| `POST /moodle/attempts/:id/review-request` | student | One per attempt, within the review period. |
| `GET /moodle/assessments/:cmid/grades?since=` | plugin task | Pending grades to write (pull, §4.4). |
| `POST /moodle/grades/ack` | plugin task | Mark grade versions as written. |
| `GET /moodle/assessments/:cmid/review-queue` | teacher | Open cases (alert / request / spot check). |
| `GET /moodle/attempts/:id/evidence` | teacher | Full conversation, scores, rationale, alerts, telemetry, feedback, times. |
| `POST /moodle/attempts/:id/decision` | teacher | Immutable decision; returns the final grade for the plugin to write. |
| `GET` / `POST /moodle/assessments/:cmid/class-report` | teacher / plugin task | Read / request generation. |
| `GET /moodle/assessments/:cmid/ux-summary` | teacher | Aggregates only (k ≥ 5). |
| `GET /moodle/assessments/:cmid/proposals`, `POST /moodle/proposals/:id/decision` | teacher | Improvement proposals and their approval (§11). |

### 4.3 — Grading on the Activity Scale

The teacher sets `grade_min` and `grade_max` (e.g. 1.0–5.0) and the decimals in the activity settings.

1. Pablo returns **points per component**, each within `[0, component_max]` of its rubric version. Code clamps out-of-range values and logs them.
2. Cycle fraction = points ÷ max points of that cycle.
3. Attempt fraction = weighted mean of cycle fractions over `total_cycles` (outcome weights, default equal). Unreached or unanswered cycles count as 0.
4. **Grade = `grade_min` + attempt fraction × (`grade_max` − `grade_min`)**, rounded to `grade_decimals`.

Steps 2–4 are deterministic backend code; the LLM never does arithmetic. The grade is always published (R3), with or without alerts. The range is locked after the first attempt starts, so published grades never change meaning. With several attempts, `grade_method` (highest / last) picks the gradebook value.

### 4.4 — Moodle Gradebook Sync (pull)

```mermaid
sequenceDiagram
    participant Plugin as mod_vervuxassess
    participant Backend as NestJS Backend
    participant GB as Moodle Gradebook

    Note over Plugin: on activity view + scheduled task (every cron run)
    Plugin->>Backend: GET /moodle/assessments/:cmid/grades?since=cursor
    Backend-->>Plugin: [{attempt, user, grade, version}]
    Plugin->>GB: grade_update(..., itemdetails: grademin, grademax)
    Plugin->>Backend: POST /moodle/grades/ack (versions)

    Note over Plugin,GB: Teacher decision (inside Moodle)
    Plugin->>Backend: POST /moodle/attempts/:id/decision
    Backend-->>Plugin: final grade (version n+1)
    Plugin->>GB: grade_update() in the same request
```

- The plugin is the **only** writer of the gradebook. Writes are idempotent by grade version.
- Moodle needs no inbound endpoint for grades: retries come free from Moodle cron, and sites behind a firewall work.
- A direct override in Moodle's gradebook bypasses the audit trail. The plugin detects Moodle's `overridden` flag and shows a warning on the review page; it never fights the override.

---

## 5 — Audio Pipeline

| Step | Detail |
|---|---|
| **Capture** | The activity UI records in the browser (MediaRecorder, opus/webm). Duration is capped per cycle (configurable). |
| **Upload** | Temporary blob to the backend (pre-signed URL). |
| **Transcription** | Pablo calls **Google Cloud Speech-to-Text v2** (synchronous `recognize`, REST + ADC, so recordings are capped at 60 s) during `agent_working`, so the timer is paused and slow transcription costs the student nothing. Language from `state.language`. |
| **Confirmation** | The transcript is shown in Pablo's acknowledgement, so the student sees what will be evaluated. |
| **Evaluation** | Pablo evaluates the **transcript**, not the audio. |
| **Deletion** | The backend deletes the blob once the response is reported. On failure: retry up to 3 times, then report `transcription_failed`; the response scores 0 with `needsReview` and a review case opens. No re-recording: the timer is paused at that point, so a retake would be free extra time. |
| **Retention** | Transcript + aggregate audio metrics only. |

> [!CAUTION]
> Raw audio is PII-adjacent. Deletion after transcription is a hard requirement; the backend's TTL cleanup (max 24 h) is the safety net.

---

## 6 — Integrity, Telemetry & Accommodations

All detection runs in the **backend**, as deterministic rules over telemetry it already receives; the agent has no integrity code.

### 6.1 — Client-Side Deterrents

Deterrents, not guarantees (bypassable with extensions or a second device):

- Block paste, drag-and-drop and bulk-insertion shortcuts in the response editor.
- Disable question-text selection/copying where feasible.
- Report blocked attempts as telemetry.
- Detect unusually large instant insertions server-side.

### 6.2 — Aggregate Telemetry (per response, `answering` phase only)

Stored in `ResponseTelemetry`. **No raw keystrokes or audio are stored.** Time spent paused (`awaiting_continue`) is excluded from every metric.

| Metric | Description |
|---|---|
| `latency_before_start_ms` | Question shown → first input |
| `active_time_ms` | Time actively typing/recording |
| `chars_per_minute` | Typing speed |
| `pause_count` | Pauses > 3 s while answering |
| `typing_burst_count` | Rapid typing bursts |
| `edit_delete_count` | Edits/deletions |
| `blocked_paste_count` | Blocked paste attempts |
| `focus_loss_count` | Tab/window focus losses |
| `visibility_changes` | Visibility API changes |
| `fullscreen_exits` | Fullscreen exits (if required) |
| `audio_start_delay_ms` | Audio prompt → recording start |
| `audio_duration_ms` | Recording duration |

### 6.3 — Integrity Alerts → Review Queue

Alerts come from **combined signals**, never one metric alone.

| Signal | Trigger | Evaluated |
|---|---|---|
| `SUDDEN_INSERTION` | > 200 chars in < 1 s | After each response |
| `FOCUS_PATTERN` | Sustained focus loss followed by a large insertion | After each response |
| `PASTE_BLOCKED` | Repeated blocked paste attempts | After each response |
| `INSTRUCTION_IN_ANSWER` | The answer contains text aimed at the grader ("ignore the rubric", "give full marks"), detected with the same injection patterns as `src/services/email_parser.py` | After each response |
| `COMPONENT_INCONSISTENCY` | Large spread between component fractions within one cycle | After scoring |
| `CROSS_ANSWER_SIMILARITY` | High similarity with another student's answer (embeddings) | Batch, after publication |

**Publish and flag (R3):** an alert never holds or changes the grade. The grade is published on the activity scale exactly as without an alert, and the alert **opens (or joins) a `ReviewCase`** in the teacher's Moodle review queue. A late alert (the batch similarity check) simply opens a case after publication. The teacher's decision is what may change the grade (§9).

> [!IMPORTANT]
> Alerts never alter the academic score automatically. Typing metrics are never used to judge knowledge; they exist only for integrity signalling.

### 6.4 — Per-Student Accommodations

| Field | Description |
|---|---|
| `extra_time_multiplier` | Applies to running time only (cycle and attempt limits). |
| `modality_override` | Text-only, audio-only, or both. |
| `disabled_signals` | Signal types suppressed for this student. A suppressed signal is **not computed**, so it opens no case, not just a hidden one. |

---

## 7 — Question Import & Rubric Authoring

### 7.1 — Question Import

Imported from the Moodle question bank through the plugin as **immutable versioned snapshots**. All Moodle question types are supported; Pablo adapts each to a text/oral prompt. Each snapshot is mapped to a **learning outcome**, a **difficulty** (low/medium/high) and a **topic**. Re-import creates a new version; in-flight and completed attempts keep theirs.

### 7.2 — Rubric Authoring

Created in the plugin UI per question or learning outcome: **components**, **criteria per score level**, **max points per component**. Every change, by hand or from an accepted proposal (§11), creates a new `RubricVersion`; an evaluation always records the version it used.

### 7.3 — Cycle Composition (Pablo)

1. Receive authorised candidates for one learning outcome.
2. Select 2–3 related questions not yet used in the attempt.
3. Compose one composite prompt with identifiable parts ("Part A … Part B …").
4. Report the cycle (prompt, component map, provenance) before it is shown.

---

## 8 — UX Survey & Academic Feedback

### 8.1 — Order

Last answer → **UX survey** → scoring → grade + feedback. Because the survey comes before scoring, the student rates the experience before seeing any result, so the rating reflects the process rather than the grade.

### 8.2 — UX Survey (fixed form, skippable)

After the last cycle (or an attempt timeout) the activity shows a form:

```
How was your experience with this assessment?
[1] [2] [3] [4] [5]      Comment (optional): [__________]
[Send]   [Skip]
```

| Rule | Detail |
|---|---|
| **Before scoring** | Scoring starts only once the survey is submitted, skipped or abandoned. |
| **Skippable** | Skip is always available (`skipped = true`). |
| **Abandoned** | If the student leaves at the survey, it is recorded `abandoned` after 10 min (configurable) or at activity close, whichever is first, and scoring proceeds, so no grade is stuck. A student who returns earlier sees the form again. |
| **No timer** | The attempt clock has stopped. |
| **Deterministic** | Rating buttons + comment field: no LLM, no parsing. |
| **Out of the transcript** | Stored in `UxSurveyResponse`, not as conversation messages, so the teacher's full-conversation view (§9.2) cannot expose an individual answer. |
| **Anonymous aggregation** | The teacher sees averages and counts only once ≥ 5 responses exist. Comments appear only as themes in the class report (§10), never verbatim per student. |
| **Zero influence** | The scoring job never receives survey data. |
| **One per attempt** | Idempotent. |

### 8.3 — Academic Feedback

Generated in the scoring job right after the grade is computed (one `generate_structured`). It is posted into the attempt's conversation and stored in `AttemptFeedback`.

| Element | Description |
|---|---|
| **Overall summary** | 2–3 sentences across all cycles. |
| **Strengths** | Tied to rubric criteria. |
| **Areas for improvement** | Concrete, constructive. |
| **Per-outcome feedback** | Score percentage + targeted recommendation per learning outcome. |
| **Study recommendations** | 2–3 actionable next steps. |

After a timeout, feedback covers the completed cycles and names the ones not reached. The teacher sees this feedback on the review page. A review decision carries its own feedback to the student, which is shown alongside Pablo's and never replaces it silently.

---

## 9 — Review Flow (entirely in Moodle)

### 9.1 — Cases and Decisions

```mermaid
stateDiagram-v2
    [*] --> Published: Scored, grade on activity scale
    Published --> CaseOpen: Integrity alert / student request / spot check
    Published --> Final: Review period expires, no case
    CaseOpen --> Decided: Teacher decides in Moodle
    Decided --> CaseOpen: Student requests review (if not yet used)
    Decided --> Final
    Final --> [*]
```

- **Sources**: `integrity_alert` (automatic), `student_request` (one per attempt, within the review period, default 7 days), `spot_check` (a random `spot_check_rate` share of unflagged attempts, default 10%, which gives §11 unbiased data). Sources open at the same time merge into one case.
- **Decision**: confirm, or set a new grade within `[grade_min, grade_max]` (optionally per component). Rationale is required; feedback to the student and the **"use as calibration example"** opt-in (§11) are optional. The decision is immutable and the latest one sets the grade.
- **A student who did not request the decided case** (alert or spot check) may still use their one request against that decision within the review period.
- Moodle sends the student a notification (Moodle messaging API) when a decision changes or confirms their grade. The student is not told about an alert unless a decision cites it.
- Capability `mod/vervuxassess:review` (editing teacher and teacher by default) gates the queue, the evidence and the decision. Teachers never need the CRM.

### 9.2 — The Review Page

Rendered server-side by the plugin (Mustache templates), with data from `GET /moodle/attempts/:id/evidence`:

| Panel | Content |
|---|---|
| **Full conversation** | Every message of the attempt in order: Pablo's composite prompts, the student's answers (text; audio as transcript, with a `transcription_failed` mark), Pablo's acknowledgements and notices, Pablo's academic feedback. Each message has its time; between cycles the pause is shown ("paused 4 min, continued 10:42"), with time used per cycle and `timed_out` marks. Source: the backend's `messages` for the attempt's conversation, not the LangGraph checkpoint. |
| **Scores** | Per component: points / max, criterion, rationale, rubric version; cycle fractions; grade on the activity scale. |
| **Integrity** | Alerts per cycle with severity and the telemetry behind each. |
| **Case** | Sources, student's reason (if requested), previous decisions. |
| **Decision form** | Moodle form (`moodleform`): confirm / new grade / per-component adjustment, rationale, feedback, "use as calibration example". On submit: backend decision, then `grade_update()` in the same request (§4.4). |

---

## 10 — Class Report (Global Feedback for the Teacher)

**Trigger:**

- **Primary:** the plugin's scheduled task sees that **every student who can attempt** (enrolled users with `mod/vervuxassess:attempt` and not `:review`, within the group if group mode is on) has at least one scored attempt, then calls `POST /moodle/assessments/:cmid/class-report`.
- **Fallback:** at the activity close date, generated with the count of students who did not finish.
- **Manual:** a "Generate now" button, which labels the report partial.
- A new attempt or decision after generation marks it **outdated**; the teacher regenerates it, and versions are kept.

**Content.** The numbers are computed in backend code; one `generate_structured` writes the narrative and themes on the tenant's model, since a person reads it.

| Section | Content |
|---|---|
| **Results** | Grade distribution on the activity scale, pass rate if `grade_pass` is set, attempts used, completion. |
| **Per learning outcome** | Mean and spread; the weakest outcomes; common misconceptions drawn from component rationales. |
| **Question analysis** | Questions with low mean or high variance across students (possibly unclear), or frequently overturned in review. |
| **Time** | Time used per cycle; cycles that often time out. |
| **Integrity** | Alerts by signal type, open and decided cases. |
| **Pablo–teacher agreement** | Share of decisions that confirmed Pablo; mean absolute difference on the activity scale (decided + spot-check cases). |
| **UX** | Average rating, response/skip rate, comment themes (k ≥ 5). |
| **Teaching recommendations** | 3–5 topics to reinforce, linked to outcomes. |
| **Improvement suggestions** | Links to the proposals this run created (§11). |

Shown on the activity's **Class report** page (capability `mod/vervuxassess:viewreports`). Students' names appear only in the results table, never in the LLM input.

---

## 11 — Self-Improvement, Human-Supervised

Pablo improves only from **human judgement**: teacher review decisions (including spot checks), teachers' accept/edit/reject on proposals, and aggregate class-report findings. Nothing changes until a human approves it.

### 11.1 — What Can Improve

| Kind | Built from | Approved by | Where | Takes effect |
|---|---|---|---|---|
| **Calibration example** | A teacher decision marked "use as calibration example" | The teacher. The opt-in itself is the approval, and it can be revoked any time. | Moodle decision form | Injected as a few-shot anchor into scoring for that rubric version (max 5 per rubric, most recent first) |
| **Rubric clarification** | ≥ 3 decisions that adjust the same component in the same direction | Teacher: accept / edit / reject | Moodle "Suggestions" page | New `RubricVersion` for new attempts |
| **Question flag** | Class-report question analysis | Teacher | Moodle "Suggestions" page | The teacher edits or retires it in the question bank and re-imports |
| **Prompt change** | Patterns across the tenant's assessments; **contains no student text** | Tenant/platform admin | Existing `ai_prompts` versioning: a **DRAFT** of `ASSESSMENT_*` → ACTIVE | New attempts |

The `insights` job (§12.2) generates the proposals. It runs after each class report and after every 10 new decisions on an assessment.

### 11.2 — Safeguards

- **Human labels only.** Pablo never learns from its own unreviewed output. Spot checks (§9.1) keep the labels from coming only from contested attempts.
- **Replay before activation.** A prompt DRAFT, and optionally a rubric version, is replayed with `POST /assessment/score` in `replay` mode (nothing published) over the assessment's decided attempts. The approver sees agreement before vs. after: mean absolute difference and the share within ±0.5 of the teacher's grade. Prompt drafts cannot be activated without a replay.
- **No retroactive change.** Published grades are never regraded by an improvement. Every `ComponentEvaluation` pins model, prompt version/sha, rubric version and calibration-set version, so any grade is reproducible.
- **Privacy scope.** Calibration examples are anonymised in code (names, emails, ids stripped), stay within their assessment (or course, if the teacher opts in), never cross tenants, never enter platform prompts, and are deleted with the attempt.
- **Drift watch.** If agreement drops after an activation (MAE up > 0.3 on the activity scale over ≥ 10 decisions), the Suggestions page and the prompt admin show a warning and a one-click **revert to the previous version**. Reverting is a human action; nothing rolls back automatically.

---

## 12 — Python Agent Changes

### 12.1 — Registry

```python
# registry.py — additive only
from .pablo_graph import build_pablo_graph

CODE_NAME_REGISTRY["pablo"] = build_pablo_graph
```

No `AGENT_TYPE_FALLBACK` entry: the fallback is phase-1 legacy for requests without `agent_code_name`, and every Pablo request comes from a backend that sends it.

### 12.2 — Graph and Jobs

- `src/graphs/pablo_graph.py`: `StateGraph(AgentState)`, nodes from §3.2, state only in `state["pablo"]`.
- `POST /assessment/score` (202, background, bounded concurrency): one `generate_structured` per cycle (all its components) + one for feedback. Answers are fenced (`<<<RESPUESTA … RESPUESTA>>>`) and the rubric comes first in the prompt. Points are clamped in code. Results go to `/internal/assessment/attempts/:id/score` and `/feedback`. On failure: one retry, then the attempt is reported `scoring_failed` and a review case opens. Never a silent zero.
- `POST /assessment/insights` (202, background): class-report narrative + proposals. Plain async functions with no checkpointer, idempotent on `report_id`.
- Models: chat turns use the tenant's `llm_model`. Scoring and feedback use the tenant's model, because students and teachers read them. Every call records `InvocationUsage` with `prompt_key/prompt_id/prompt_version/prompt_sha` and `latency_ms` (as clara does).

### 12.3 — Files

```
src/graphs/pablo_graph.py
src/agents/pablo/
├── __init__.py
├── common.py          # prompt resolution + provenance, fences, stream events
├── jobs.py            # credentials, de-duplication and concurrency for the background jobs
├── nodes.py           # pablo_triage, pablo_next_cycle, pablo_receive_answer, pablo_timeout, pablo_error
├── compose.py         # question selection + composite prompt
├── scoring.py         # scoring job: per-cycle structured scoring, clamping, calibration examples
├── feedback.py        # academic feedback
├── insights.py        # class report narrative + improvement proposals
├── prompts.py         # ASSESSMENT_COMPOSE / _SCORE / _FEEDBACK / _CLASS_REPORT / _IMPROVE (mirrored in backend DEFAULT_PROMPTS)
├── texts.py           # fixed Spanish messages: acknowledgement, last question, timeout, holding
└── transcription.py   # Google Cloud STT v2
```

Removed from the previous plan: `ux_survey.py` (form owned by backend + plugin), `timeout.py` (backend clock), `integrity.py` (backend rules).

### 12.4 — Migrations

Backend (Prisma): `pablo` row in `agent_code_names`, `ASSESSMENT` agent type, the §4.1 tables, `Conversation.attemptId`, and the `ASSESSMENT_*` default prompts. The agent needs no migration: checkpoints and the store already exist.

---

## 13 — Moodle Plugin (`mod_vervuxassess`)

- **Type**: activity module, with `$plugin->dependencies = ['local_vervuxchat' => …]` to reuse its JWT signing/verification. It has **its own** channel connection credentials (key + identity secret, encrypted backend-side as for MOODLE), separate from Ismael's.
- **Capabilities**: `addinstance`, `view`, `attempt` (student), `review` (teacher, editing teacher), `viewreports`, `manageimprovements`.
- **Pages**: settings (`mod_form`: dates, durations, attempts, grade method, **grade min/max/decimals**, pass grade, review period, modality, spot-check rate); question import and mapping; rubrics; accommodations; attempt UI; review queue; review page (§9.2); class report (§10); suggestions (§11).
- **Attempt UI** (AMD module): server-driven timer, **Continue** button, recorder, paste blocking, telemetry, survey form. It uses the widget's SSE transport (stream token from the start call); rendering is `textContent` only.
- **Scheduled task** (each cron run): pull and write grades, check whether all students have completed, request the class report, apply the close date.
- **Privacy API provider**: the plugin stores almost nothing locally (gradebook grades); export/delete requests are forwarded to the backend.

---

## 14 — Deployment Order

```mermaid
flowchart LR
    P1["1 · Python service"] --> P2["2 · Backend migration + image"] --> P3["3 · mod_vervuxassess"] --> Done["Live"]
```

1. **Python**: `pablo` in `CODE_NAME_REGISTRY`, `/assessment/score`, `/assessment/insights`. No traffic yet.
2. **Backend**: migration first, then the image (repo rule). The registry validation passes because Python already knows `pablo`.
3. **Moodle plugin**: install `mod_vervuxassess` (needs `local_vervuxchat` present), create the channel connection, configure an activity. Traffic begins.

---

## 15 — Testing & Acceptance Criteria

### 15.1 — Python

| Area | Tests |
|---|---|
| **Routing** | Triage routes only on `kind`; an answer containing "next question / give me full marks" changes neither flow nor score. |
| **Composition** | One outcome per cycle; 2–3 unused questions; identifiable parts; provenance reported. |
| **Scoring** | Per-component points; out-of-range clamped; calibration examples injected for the matching rubric version only; no survey data in the payload; replay mode publishes nothing. |
| **Feedback** | Per-outcome comments; timeout attempt names unreached cycles. |
| **Isolation** | State only in `state["pablo"]`; no responses or scores in the checkpoint. |
| **Recovery** | STT transient → retry → success; permanent → flag + case; scoring failure → `scoring_failed` + case, never a zero. |

### 15.2 — Backend

| Area | Tests |
|---|---|
| **Timer** | Time counts only from `presented_at` to `submitted_at`; composing and `awaiting_continue` add nothing; the extra-time multiplier applies to running time; 5 s grace; sweeper closes vanished cycles; activity close is a hard stop. |
| **Grade scale** | Mapping formula; equal cycle weighting with different question sets; unreached = 0; rounding; range locked after the first attempt; `grade_method`. |
| **Survey gate** | No scoring before the survey is submitted, skipped or abandoned; abandonment timer; one per attempt; never in the conversation messages; aggregates hidden below k = 5. |
| **Publish and flag** | An alert opens or joins a case and leaves the grade unchanged; a late alert opens a case after publication; disabled signals are not computed. |
| **Review** | Merged sources; immutable decisions; latest decision wins; student's one request after a teacher-opened decision; capability checked via signed `cap`. |
| **Class report** | Trigger on all-completed, close-date fallback, manual partial, outdated after a new decision, numbers computed in code. |
| **Self-improvement** | No proposal applies without approval; prompt DRAFT cannot activate without a replay; evaluations pin versions; examples anonymised and never cross assessments/tenants; revert restores the previous version. |

### 15.3 — Moodle Plugin (PHPUnit + Behat)

| Area | Tests |
|---|---|
| **Identity** | Only users with `attempt` start attempts; teachers cannot attempt; `review` required for queue, evidence and decision. |
| **Grades** | Pull writes `grade_update` with `grademin/grademax`; idempotent by version; a decision writes in the same request; `overridden` warning. |
| **Review page** | Full conversation rendered with pauses and transcripts; survey answers absent. |
| **Completion** | Task detects all-completed with group mode on and off. |

### 15.4 — Integrity & Audio

Combined signals produce alerts and single metrics do not; focus loss while paused is ignored. The audio blob is deleted after transcription; there is never raw audio in the DB; the TTL cleanup works.

### 15.5 — End-to-End

1. Teacher creates the activity: grade range 1.0–5.0 (1 decimal), 3 cycles, max attempts 2, modality "both", spot checks 10%.
2. Teacher imports questions, maps outcomes, writes rubrics, gives Student B 1.5× time and disables `FOCUS_PATTERN`.
3. Student A answers cycle 1 (text). The timer pauses; Pablo acknowledges; A waits 3 min and presses Continue; cycle 2 appears and the timer resumes. The 3 min are not counted.
4. Student A answers cycle 3 by audio and sees the transcript in the acknowledgement.
5. The survey appears before any grade. A rates 4/5 with a comment about time pressure.
6. Pablo scores; the backend maps to e.g. 3.8; the plugin writes it to the gradebook; A reads the academic feedback.
7. Student C pastes a large block in cycle 2: `SUDDEN_INSERTION`. C's grade is still published; a case appears in the teacher's queue.
8. The teacher opens C's case in Moodle, reads the full conversation (pauses, answers, Pablo's messages), lowers the grade with a rationale and ticks "use as calibration example". The gradebook updates immediately and C is notified.
9. Student A requests a review; the teacher confirms 3.8 in Moodle.
10. Student B finishes with extended time and skips the survey; scoring proceeds.
11. When the last enrolled student finishes, the class report appears in Moodle: distribution, weak outcomes, a flagged question, agreement figures, UX themes (≥ 5 responses).
12. The Suggestions page proposes a rubric clarification; the teacher edits and accepts it, creating a new rubric version for new attempts only. A's and C's grades are untouched.
13. Student A's attempt 2 is scored with the new rubric version and C's calibration example; its evaluation records both versions.

---

## 16 — Assumptions & Constraints

1. Pablo is a new activity module (`mod_vervuxassess`), not a quiz question type, not part of the chat bubble.
2. All teacher work happens in Moodle: configuration, review decisions, class report and approval of course-level improvements. Only tenant/platform prompt activation uses the CRM's existing prompt versioning.
3. The chat graph only talks to students; teacher-facing outputs are background jobs.
4. Pablo and Ismael run on separate channel connections per tenant.
5. Identity and capabilities are server-attested by the plugin (signed JWT), never taken from the browser.
6. The backend owns the clock, enrolment checks, grade mapping and integrity rules; the LLM owns composition, component points, rationale and narrative.
7. Grades are always published on the activity's `[grade_min, grade_max]` scale; alerts flag, never hold or change.
8. The UX survey precedes scoring and is always skippable; it never reaches the scoring job.
9. Self-improvement uses only human-reviewed data and only applies after human approval; nothing is retroactive.
10. Deploy order: Python → Backend → Moodle plugin.

---

## 17 — Open Questions & Risks

| # | Question / Risk | Proposed default |
|---|---|---|
| 1 | **No cap on the Continue pause.** A student could leave the attempt paused for days. | Bounded by the activity close date. Add an optional `max_pause_minutes` if institutions ask. |
| 2 | **"All students completed" may never happen** (students who never open the activity). | Close-date fallback + "Generate now". |
| 3 | **Spot-check workload.** 10% of attempts adds teacher reviews. | Configurable per activity; 0 turns it off (the improvement loop then relies on contested cases only, with a warning about bias). |
| 4 | **Course backup/restore and import.** Moodle reuses courses each term; outcomes, rubrics and snapshots live in the backend. | V1: restore creates an unconfigured activity, and the plugin offers "copy configuration from the original" via the backend. Full backup/restore support in V2. |
| 5 | **Gradebook overrides** bypass the audit trail. | Warn on the review page; document it for teachers. |
| 6 | **Students are not told about alerts.** Some institutions may require disclosure. | Off by default; a tenant setting could notify on case opening. |
| 7 | **STT latency** with long audio. | The timer is paused during transcription; cap audio duration per cycle. |
| 8 | **Question-type fidelity** (matching, drag-and-drop) when adapted to text prompts. | Document per-type limitations; the class report's question analysis surfaces items that perform badly. |
| 9 | **Cross-answer similarity cost** (O(n²)). | Embedding index, batch after publication, threshold-only cases. |
| 10 | **Audio deletion compliance.** | 24 h TTL cleanup + an alert on orphaned blobs. |
| 11 | **Crash mid-attempt.** | `GET /moodle/attempts/:id/state` resumes the phase and the remaining time; the thread persists. |
| 12 | **Calibration examples drifting a rubric** toward one teacher's style. | Max 5 per rubric, revocable, replay-checked; drift watch (§11.2). |
| 13 | **Survey response rate** (skippable). | Frictionless form; low rates accepted; the class report shows the response rate next to the averages. |
