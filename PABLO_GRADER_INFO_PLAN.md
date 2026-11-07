# Pablo usa «Información para evaluadores» (calificación, veredicto, repregunta y retroalimentación)

> Este plan también incluye las **correcciones del intento** surgidas en las pruebas de producción
> del 25 al 27-oct (sección 4): el profesor elige cuántas preguntas tiene cada ciclo, el banco
> agotado, el «Escribiendo…» sin fin, la vista previa fuera de fechas y la interfaz. Todo se
> despliega junto.

## Contexto

Hoy Pablo solo recibe del banco el **texto de la pregunta** (`question_bank::snapshot()` →
`AssessmentQuestion.text` → `scoringPayload`). La respuesta esperada que el profesor escribe en
**Información para evaluadores** (`qtype_essay_options.graderinfo`, solo en preguntas Ensayo) no
llega al backend ni al agente, de modo que Pablo evalúa únicamente con los descriptores de la
rúbrica sin saber cuál es la respuesta conceptualmente correcta esperada por el docente.

Decisiones de diseño (acordadas con el usuario):

1. **Calificación (verdad base para la rúbrica + veredicto por pregunta)**:
   - Pablo evalúa **primero** la exactitud conceptual de la respuesta del estudiante frente a
     **Información para evaluadores** (`grader_info`) y emite por cada pregunta un **veredicto**
     (`correct` / `partial` / `incorrect`) con su `reason` técnica para el profesor.
   - Ese juicio de exactitud actúa como **verdad base** al calificar cada componente de la rúbrica:
     si el estudiante contradice u omite lo esencial de la referencia, no puede alcanzar niveles
     altos en criterios de comprensión/contenido aunque redacte bien. La escala de puntos (`0` a
     `max_points`) sigue mandándola la rúbrica, ignorando puntajes escritos dentro de
     `grader_info`.
   - En el esquema `CycleScore`, `verdicts` va **antes** de `components` para forzar que el
     modelo razone primero sobre la correctitud frente a la referencia y luego asigne los puntos.
2. **Retroalimentación al estudiante (desglose por pregunta + orientación conceptual sin filtrar la solución)**:
   - Además del resumen global (`overall_summary`, `strengths`, `areas_for_improvement`,
     `per_outcome_feedback`, `study_recommendations`), la retroalimentación incluye un **desglose
     por pregunta** (`per_question_feedback`) que indica al estudiante si cada pregunta/parte fue
     **Correcta**, **Parcialmente correcta** o **Incorrecta** (o solo el comentario si la pregunta
     no tiene referencia) junto con retroalimentación específica.
   - **Orientación conceptual sin revelar la respuesta**: cuando el estudiante falla o responde a
     medias, Pablo le explica qué parte de su razonamiento estuvo bien, dónde estuvo el error o
     vacío conceptual y qué tema debe repasar, **sin citar, parafrasear ni entregar la respuesta
     esperada**, protegiendo el banco de preguntas.
3. **Repregunta en vivo (`followup`) guiada por la referencia**:
   - Durante la conversación, `followup` recibe la referencia de las partes del ciclo para
     detectar si la respuesta inicial es incompleta o conceptualmente errónea frente a lo esperado
     y enfocar la repregunta en ese punto, sin dar pistas ni emitir juicios de valor y con el
     guardia léxico (`leaked_words`) ampliado sobre la referencia.
4. **Alcance del campo**:
   - Solo `qtype_essay_options.graderinfo` (preguntas tipo Ensayo). Una pregunta sin ese campo se
     califica solo con la rúbrica y su veredicto queda `no_reference` («Sin referencia»).
5. **También para el profesor**:
   - En la revisión del intento (bloque plegable con «Información para evaluadores» + insignia de
     veredicto y `reason`) y en el informe del grupo (`misconceptions` contrastados con la
     referencia y `question_flags` apoyados en la distribución de veredictos).

Regla transversal de seguridad: **El texto de «Información para evaluadores» nunca llega al
estudiante.** No lo ven ni el compositor (`compose.py`) ni el acuse (`echo.py`), ni se expone en
los endpoints de vista del estudiante. Solo entra, cercado con `<<<REFERENCIA … REFERENCIA>>>`, a:
- calificación (`scoring.py`),
- repregunta (`followup.py`, protegida por `leaked_words`),
- retroalimentación (`feedback.py`, protegida por `copied_span`),
- vistas del profesor e informe de grupo.

---

## Ramas

| Repo | Rama nueva | Desde |
|---|---|---|
| `yorchio-backend` | `feat/assessment-grader-info` | `release/grade-average` (producción) |
| `yorchio-agent` | `feat/pablo-grader-info` | `main` (`86ac1b2`, producción) |
| `yorchio-moodle` | `feat/assessment-grader-info` | `feat/assessment-grade-average` (0.3.2) |

---

## 1 · Plugin (`yorchio-moodle/vervuxassess/`, 0.3.3)

- `classes/question_bank.php`
  - `snapshot()`: si `qtype === 'essay'`, lee `graderinfo, graderinfoformat` de
    `qtype_essay_options` (por `questionid` de la última versión lista), lo pasa por
    `content_to_text()` igual que el enunciado, aplica `trim` y lo recorta a 20.000 caracteres.
    Lo devuelve como `graderInfo` (vacío → se omite).
  - `card()` y las consultas de `search()`/`resolve()`: `LEFT JOIN {qtype_essay_options}` para
    agregar `reference: bool`. El editor (`amd/src/setup_editor.js`, chips de pregunta) muestra
    «Con referencia» o «Sin referencia»; no bloquea.
- `classes/setup.php::readiness()`: aviso informativo, no bloqueante: «N preguntas sin
  Información para evaluadores: Pablo las califica solo con la rúbrica».
- Revisión del profesor (`attempt.php` + `templates/review_attempt.mustache`): por cada pregunta
  del ciclo, un bloque plegable «Información para evaluadores» y una insignia con el veredicto
  (`Correcta` / `Parcialmente correcta` / `Incorrecta` / `Sin referencia`) y su motivo (`reason`),
  ambos leídos desde `evidence`.
- Vista del estudiante: el estudiante recibe en el mensaje de cierre de retroalimentación (y en
  el objeto `feedback.perQuestionFeedback`) el veredicto y el comentario formativo de cada
  pregunta, nunca `graderInfo` ni el `reason` interno del profesor.
- `db/upgrade.php`: paso nuevo que encola `\mod_vervuxassess\task\sync_setup` para cada
  instancia existente, de modo que las actividades ya creadas sincronicen `graderInfo` sin
  obligar al profesor a volver a guardar.
- Strings `es`/`en`: `reference_yes`, `reference_none`, `readiness_noreference`,
  `graderinfo`, `verdict_correct|partial|incorrect|no_reference`.
- `version.php`: `0.3.3`.

---

## 2 · Backend (`yorchio-backend/src/assessment/`)

- **Migración aditiva** `2026102800000_assessment_grader_info`:
  - `assessment_questions."graderInfo" TEXT NULL`;
  - tabla `assessment_question_verdicts` con `id`, `attemptId` (FK cascade), `scoringVersion`,
    `cycleIndex`, `questionSnapshotId`, `verdict VARCHAR(16)`, `reason TEXT`, `createdAt`, y
    `@@unique([attemptId, scoringVersion, cycleIndex, questionSnapshotId])`.
- `dto/assessment.dto.ts`:
  - `QuestionDto.graderInfo?` (opcional, `≤ 20000`; compatible con plugins anteriores);
  - `ReportScoreDto.verdicts?: QuestionVerdictDto[]`, con `cycleIndex`, `questionSnapshotId`,
    `verdict ∈ {correct, partial, incorrect, no_reference}` y `reason ≤ 1000`;
  - `AttemptFeedbackDto.perQuestionFeedback?: PerQuestionFeedbackDto[]`, con `cycleIndex`,
    `questionSnapshotId`, `part` (`≤ 40`), `verdict ∈ {correct, partial, incorrect, no_reference}`
    y `comment` (`≤ 1000`).
- `assessment-config.service.ts` (upsert de la pregunta, ~l. 245): guarda `graderInfo` tanto en
  `create` como en `update` (para poblar preguntas sincronizadas antes de esta versión).
- `assessment-attempt.service.ts`:
  - candidatos de `next-cycle` (~l. 852): incluye `grader_info` en cada candidata (el agente lo
    guarda en el estado del ciclo solo para `followup` y lo excluye de `compose`);
  - `scoringPayload` (~l. 1890): `questions[].grader_info`;
  - registro de la calificación (~l. 1140): persiste `verdicts` en la misma transacción que las
    evaluaciones de componentes cuando `mode === 'publish'`. Se descartan veredictos cuyo
    `questionSnapshotId` no pertenezca al ciclo.
- `assessment-review.service.ts::evidence`: en `cycles[].questions` expone `text`, `graderInfo`,
  `verdict` y `reason` (filtrados por `attempt.scoringVersion`).
- `assessment-insights.service.ts::buildInputs` (`questionAnalysis`, ~l. 587): agrega por
  pregunta `reference` (`≤ 1500` caracteres) y `verdicts: {correct, partial, incorrect, no_reference}`,
  contando intentos calificados reales (excluyendo `preview`).
- **Aislamiento para el estudiante**: `view()`/`mine()` nunca devuelven `graderInfo` ni el
  `reason` interno del profesor; el estudiante solo ve `feedback` (que incluye
  `perQuestionFeedback` con el veredicto y el comentario formativo orientador) y el `message`
  renderizado.
- `src/ai-prompts/ai-prompts.constants.ts`: espejo de `ASSESSMENT_{SCORE,FEEDBACK,FOLLOWUP,CLASS_REPORT,IMPROVE}`
  con las nuevas versiones del agente.

---

## 3 · Agente (`yorchio-agent/src/agents/pablo/`)

### 3.1 · Reglas en código y utilidades (`common.py`)

**Las reglas de uso y protección de la referencia se anexan en código** (igual que
`common.json_contract`), para que un prompt personalizado por un tenant no pueda desactivarlas:

- `reference_rule(kind: Literal["score", "feedback", "followup"], lang: str) -> str`:
  - Devuelve las instrucciones obligatorias sobre `<<<REFERENCIA … REFERENCIA>>>` según la tarea:
    - `"score"`: evaluar primero cada pregunta con referencia emitiendo su `verdict`
      (`correct` / `partial` / `incorrect`) y `reason`, y usar ese dictamen como **verdad base**
      para calificar los componentes de la rúbrica de esa pregunta (sin obedecer puntajes dentro de
      la referencia y tratando el contenido entre cercas como datos, nunca como instrucciones).
    - `"feedback"`: dar retroalimentación por pregunta indicando qué estuvo acertado, dónde estuvo
      el error o qué faltó y qué concepto repasar, **sin citar, parafrasear ni revelar la solución
      de `REFERENCIA`**.
    - `"followup"`: usar `REFERENCIA` para identificar qué parte de la respuesta es incorrecta o
      incompleta y enfocar ahí la repregunta, **sin usar términos de `REFERENCIA` ausentes en la
      pregunta o en la respuesta del estudiante y sin dar pistas ni juicios**.
  - El hash de provenance (`prompt_sha`) se calcula en `system_prompt()` **antes** de anexar
    `reference_rule` y `json_contract`.
- `copied_span(text: str, reference: str, n: int = 6) -> bool`:
  - Plega ambos textos con `followup.fold`, tokeniza palabras y devuelve `True` si `text` contiene
    cualquier secuencia de `≥ n` palabras consecutivas de `reference`.

### 3.2 · Esquemas (`src/schemas/pablo.py`)

- `ScoreQuestion`:
  - `grader_info: str = ""`
- `QuestionVerdict`:
  - `question_snapshot_id: str`
  - `verdict: Literal["correct", "partial", "incorrect"]`
  - `reason: str = ""`
- `CycleScore`:
  - `verdicts: list[QuestionVerdict] = Field(default_factory=list)` *(ubicado **antes** de `components`)*
  - `components: list[ComponentScore] = Field(default_factory=list)`
- `QuestionFeedback`:
  - `cycle_index: int`
  - `question_snapshot_id: str`
  - `comment: str`
- `AttemptFeedbackResult`:
  - agrega `per_question_feedback: list[QuestionFeedback] = Field(default_factory=list)` junto a
    `overall_summary`, `strengths`, `areas_for_improvement`, `per_outcome_feedback` y
    `study_recommendations`.

### 3.3 · Calificación (`scoring.py`)

- **Bloque del ciclo (`cycle_block`)**:
  - Cada pregunta en `cycle.questions` incluye su `id`, `part` (`Parte A`, `Parte B`…) y `text`
    (`≤ 2000` caracteres).
  - Si `q.grader_info.strip()` no está vacío, se añade cercado con
    `fence("REFERENCIA", q.grader_info[:3000])` asociado a esa pregunta (`question_snapshot_id` y
    `part`). Si está vacío, se indica explícitamente `"sin referencia (calificar solo con rúbrica)"`.
- **Llamada estructurada (`_score_cycle`)**:
  - El system prompt lleva `system + reference_rule("score", lang) + json_contract(CycleScore)`.
  - El modelo produce primero `verdicts` (uno por pregunta con referencia en el ciclo) y luego
    `components` usando esos veredictos como verdad base para aplicar los criterios de la rúbrica.
- **Validación y normalización de veredictos en código (`verdicts_from(cycle, result)`)**:
  - Por cada pregunta `q` en `cycle.questions`:
    - Si `not q.grader_info.strip()`: se emite determinísticamente
      `{"cycleIndex": cycle.cycle_index, "questionSnapshotId": q.question_snapshot_id, "verdict": "no_reference", "reason": ""}`
      (el modelo no decide `no_reference`).
    - Si tiene `grader_info` y el modelo devolvió un `QuestionVerdict` válido (`correct`,
      `partial` o `incorrect`) para ese `question_snapshot_id`: se guarda con `reason[:1000]`.
    - Si el modelo omitió el veredicto de una pregunta con `grader_info` o devolvió un valor
      inválido: se lanza `ScoringFailed` para que opere el reintento de `_score_cycle` (igual que
      cuando falta un componente), garantizando que toda pregunta respondida con referencia tenga
      su veredicto.
  - Cualquier `question_snapshot_id` inventado por el modelo que no pertenezca al ciclo se
    descarta.
- **Ciclos sin respuesta o con transcripción fallida**:
  - Si el ciclo no tiene respuesta (`not answer_text(cycle)`):
    - Si `transcription_status == "failed"`, no se emiten veredictos (queda para revisión del
      profesor).
    - Si el estudiante dejó el ciclo en blanco, para las preguntas con `grader_info` se emite
      `verdict: "incorrect"` con `reason: _NO_ANSWER[lang]` (y `"no_reference"` para las que no
      tengan `grader_info`), sin llamada al modelo.
- **Reporte al backend (`scoring.run`)**:
  - En `mode == "publish"` envía `verdicts` dentro del payload de `report_assessment_score` y los
    pasa a `feedback.run(req, config, rows, verdicts)`.
  - En `mode == "replay"` no envía `verdicts` ni ejecuta `feedback`.

### 3.4 · Retroalimentación al estudiante (`feedback.py`)

- **Entrada (`feedback_block(req, rows, verdicts, outcomes, include_references=True)`)**:
  - Por cada ciclo y cada pregunta (`part`, `question_snapshot_id`):
    - Muestra el texto de la pregunta compuesta, la respuesta del estudiante cercada con
      `fence("RESPUESTA", ...)`, los componentes calificados y, por cada pregunta:
      - su `verdict` (`correct` / `partial` / `incorrect` / `no_reference`) y el `reason` de la
        calificación;
      - si `include_references=True` y la pregunta tiene `grader_info`, su bloque
        `fence("REFERENCIA", q.grader_info[:2000])`.
- **Contrato y reglas (`FEEDBACK_PROMPT` + `reference_rule("feedback", lang)`)**:
  - Pide poblar `per_question_feedback` con un elemento por cada pregunta de los ciclos
    respondidos (`cycle_index`, `question_snapshot_id`, `comment`).
  - En `comment`:
    - Si la respuesta fue `correct`: confirmar qué conceptos clave explicó bien.
    - Si fue `partial` o `incorrect`: señalar qué parte de su razonamiento fue válida, explicar
      con claridad dónde estuvo la confusión o qué aspecto conceptual faltó abordar y recomendar
      qué tema o concepto repasar, **sin revelar, citar ni parafrasear la respuesta esperada de
      `REFERENCIA`**.
- **Ensamblaje determinista y renderizado (`render`)**:
  - El código cruza `result.per_question_feedback` contra las preguntas reales del intento y
    adjunta de forma determinista `part` y `verdict` (desde `verdicts`, nunca inventado por el
    modelo). Si una pregunta fue dejada en blanco, el código completa su entrada con el mensaje de
    sin respuesta.
  - En `render()` (el mensaje que se publica en el chat del estudiante) se agrega la sección
    **«Por pregunta»** / **«By question»** antes de «Por resultado de aprendizaje»:
    - Con referencia: `- Pregunta {n} ({part}) — {Correcta | Parcialmente correcta | Incorrecta}: {comment}`
    - Sin referencia: `- Pregunta {n} ({part}): {comment}`
  - En el JSON enviado a `report_assessment_feedback`, `feedback["perQuestionFeedback"]` incluye
    `[{cycleIndex, questionSnapshotId, part, verdict, comment}]`.
- **Guardia anti-filtración (`copied_span`)**:
  - Tras generar y renderizar la retroalimentación (`message` + todos los campos de texto de
    `AttemptFeedbackResult`), se verifica `copied_span(full_text, q.grader_info, n=6)` contra
    todas las referencias del intento.
  - Si detecta copia:
    1. Reintenta **una vez** agregando al prompt de usuario una advertencia explícita de no copiar
       ni parafrasear frases de `REFERENCIA`.
    2. Si en el reintento vuelve a dispararse `copied_span`, hace una llamada de respaldo con
       `include_references=False` (conservando los veredictos y las justificaciones de rúbrica, pero
       sin pasar `grader_info`), registra el log estructurado `pablo_feedback_reference_leak` e
       incrementa el contador Prometheus correspondiente.

### 3.5 · Repregunta en vivo (`compose.py`, `nodes.py`, `followup.py`)

- `compose.py`:
  - `candidates_block()` mantiene su lista blanca actual (`question_snapshot_id`, `type`,
    `difficulty`, `topic`, `text`) — **`grader_info` nunca entra al compositor**.
  - Nueva función `question_grader_info(parts, candidates) -> dict[str, str]`: devuelve
    `{part.question_snapshot_id: str(cand.get("grader_info") or "").strip()[:2000]}` para las
    partes elegidas que tengan referencia.
- `nodes.py` (`pablo_next_cycle_node`):
  - Guarda `"grader_info": compose.question_grader_info(parts, candidates)` dentro de `cycle` en
    `state["pablo"]` (junto a `rubric_criteria`; no se envía en `report_assessment_cycle`).
- `followup.py`:
  - `plan_for()` anexa `reference_rule("followup", lang)` al system prompt y, si `cycle.get("grader_info")`
    tiene entradas no vacías, añade al mensaje de usuario las referencias de cada parte cercadas
    con `fence("REFERENCIA", ...)`.
  - Así el modelo identifica qué parte/componente presenta un vacío o error conceptual frente a la
    referencia (`target_component_id`, `mode: "gap"`) y formula una repregunta abierta que invite
    al estudiante a reconsiderar o profundizar ese punto sin darle pistas.
  - **Guardia léxico ampliado en `validated()`**:
    - Combina el texto de la rúbrica y todas las referencias del ciclo:
      `restricted = f"{rubric} {' '.join((cycle.get('grader_info') or {}).values())}"`.
    - `leaked_words(question, restricted, f"{cycle.get('composite_prompt', '')} {answer}")`:
      cualquier palabra de contenido (`≥ 5` letras, plegada, fuera de `_GENERIC`) presente en la
      rúbrica **o en `grader_info`** que no haya aparecido ni en la pregunta mostrada ni en la
      respuesta del estudiante rechaza la repregunta (`pablo_followup_leak_refused`) y envía la
      repregunta genérica (`generic_question`).

### 3.6 · Informe del grupo y propuestas de mejora (`insights.py`)

- `CLASS_REPORT_PROMPT` y `report_block()`:
  - `req.question_analysis` ya viaja en `report_block()`; ahora incluye `reference` y el conteo de
    `verdicts` por pregunta.
  - `misconceptions` contrasta los errores recurrentes del grupo frente a `reference` (sin citar a
    ningún estudiante).
- `IMPROVE_PROMPT`:
  - `question_flags` utiliza la distribución de `verdicts` (p. ej. alta proporción de `incorrect`
    o discrepancia entre los puntos de rúbrica y el veredicto frente a la referencia) para señalar
    preguntas ambiguas o rúbricas desalineadas con la «Información para evaluadores».

### 3.7 · Prompts y documentación

- `src/agents/pablo/prompts.py`:
  - Actualizar `SCORE_PROMPT`, `FEEDBACK_PROMPT`, `FOLLOWUP_PROMPT`, `CLASS_REPORT_PROMPT` e
    `IMPROVE_PROMPT` y subir sus `*_VERSION` a `"2026-10-28"`.
- Documentación:
  - Actualizar la sección `### Assessment agent (pablo, agent type ASSESSMENT)` en
    `yorchio-agent/CLAUDE.md`.

---

## 4 · Correcciones del intento (incidentes del 25 al 27-oct)

Abarcan los tres repositorios. Van en las mismas ramas que el resto del plan y se despliegan con
él (decisión del usuario).

### 4.1 · Qué pasó en producción

Datos leídos de `vervux-platform-prod` el 2026-10-27, de los logs y de la base (túnel IAP, solo
lectura). Todas las pruebas son del usuario de Moodle 3739, desde `aulas.unidadeducativa.org`.

| Actividad | Configuración | Intento | Estado |
|---|---|---|---|
| 86226 | 3 preguntas, 1 resultado, `totalCycles = 3`, `maxAttempts = 1`, sin fechas, no conversacional | `d2012350` (real, 25-oct) | `retry_compose` en el ciclo 2. El ciclo 0 usó 2 preguntas y el 1 la única que quedaba. El 2 no tiene ninguna (`no_candidates`, dos reintentos fallidos). Ocupa el único intento de ese usuario |
| 86226 | | `4d0be45c` (vista previa, 26-oct) | `answering`, ciclo 0 con 2 preguntas. Funciona bien |
| 86235 | 3 preguntas, 1 resultado, `totalCycles = 3`, `maxAttempts = 2`, abría el 27-oct a las 12:00 UTC, conversacional | 7 × `POST /attempts` (02:40–03:12 UTC) | 403 `NOT_OPEN` → «Escribiendo…» sin fin (4.4) |
| 86235 | | `ec7cdd5d` (27-oct 03:26 UTC) | Se pulsó `/begin` en vez del calentamiento (4.6). El ciclo 0 usó las 3 preguntas, así que el ciclo 1 fallará con `no_candidates` |

Causas:

1. **Las preguntas por ciclo (2–3) varían y el profesor no lo ve.** El agente elige 2 o 3 por
   ciclo (`compose.MIN_PARTS`/`MAX_PARTS`). La comprobación del plugin
   (`setup.php::readiness`, `'enough' => $questions >= $cycles`) supone 1. Con 3 preguntas y 3
   ciclos la actividad sale como «lista» y siempre se atasca en el ciclo 1 o en el 2. Ver 4.2 y 4.3.
2. **El plugin se queda en «Escribiendo…» si falla el primer `POST /attempts`.** Ver 4.4.
3. **La vista previa del profesor respeta las fechas**: no se puede probar antes de abrir. Ver 4.4.
4. **El 403 no deja rastro**: hizo falta abrir la base para saber que era `NOT_OPEN`. Ver 4.4.
5. **El calentamiento se ofrece como lo normal («Puedes saltarla»)**, pero el botón principal es
   «Empezar con las preguntas» y el del calentamiento es secundario. Ver 4.6.

### 4.2 · El profesor elige cuántas preguntas tiene cada ciclo

Decisiones (usuario):

- `questionsPerCycle` (`k`) ∈ {1, 2, 3}, **1 por defecto**. Las actividades que ya existen quedan
  en 1, así que 86226 y 86235 (3 preguntas para 3 ciclos) pasan a ser válidas.
- **Si el banco no alcanza, no se puede empezar el intento.**
- **Se cuenta por resultado.** Un ciclo nunca mezcla preguntas de otro resultado.

Regla: con `n` resultados activos (ordenados por `position`) y `M` ciclos, el resultado `j`
recibe `c_j = ceil((M − j) / n)` ciclos (0 si `j ≥ M`), por la rotación de `outcomeForCycle`
(ciclo `i` → resultado `i % n`). Hace falta que el resultado `j` tenga al menos `k × c_j`
preguntas elegibles (activas y con rúbrica).

Plugin (0.3.3):

- `mod_form.php`: un select `questionspercycle` (1/2/3, por defecto 1) junto a `totalcycles`, con
  ayuda: «Cuántas preguntas del banco junta Pablo en cada turno. Ajusta el tiempo por ciclo en
  consecuencia».
- `db/install.xml` + `db/upgrade.php`: campo `questionspercycle` `INT(2) NOT NULL DEFAULT 1`.
  Incluirlo en la copia de seguridad y la restauración (`backup/moodle2/`).
- `classes/setup.php`: sincroniza `questionsPerCycle`. En `readiness()`, `'enough'` pasa a usar la
  regla por resultado y le detalla al profesor lo que falta («La fe: necesita 3, tiene 2»).

Backend:

- Migración: `assessments."questionsPerCycle"` y `assessment_attempts."questionsPerCycle"`, ambos
  `INT NOT NULL DEFAULT 1`. El intento guarda una copia del valor al crearse, igual que
  `totalCycles`, para que un cambio de configuración no altere un intento en curso.
- DTO de configuración: `questionsPerCycle?`, `@IsInt() @Min(1) @Max(3)`. Si un plugin anterior no
  lo envía, se conserva el valor guardado.
- `assessment-grade.ts`: `cyclesForOutcome(totalCycles, outcomeCount, j)`, junto a
  `outcomeForCycle`, y `bankShortfall(...)` → `[{outcomeId, name, need, have}]`.
- `start()` (~l. 260, después de `NOT_CONFIGURED`): si falta alguna pregunta, responde
  `409 NOT_ENOUGH_QUESTIONS` con `missing` en el cuerpo, también en la vista previa (así el
  profesor ve qué falta), y deja log (4.4).
- `next-cycle` (~l. 818): se quita el respaldo «outcome bank spent», que tomaba preguntas de
  cualquier resultado. Envía `questions_per_cycle: attempt.questionsPerCycle`.
  `MAX_CANDIDATES = 8` sigue sirviendo (≥ 3).

Agente:

- `compose.validated_parts(result, candidates, k)`: acepta hasta `k` partes del modelo y completa
  con candidatas tal cual hasta `min(k, len(candidates))`. Si el payload no trae
  `questions_per_cycle` (backend anterior), sigue con 2–3 como hoy.
- `COMPOSE_PROMPT`: la regla «Elige entre 2 y 3…» sale del texto que el tenant puede editar y se
  añade en código (`compose.count_rule(k, lang)`, igual que `json_contract`): «Elige exactamente
  k candidatas…», o «Usa una sola candidata» cuando `k = 1`. Se sube `COMPOSE_VERSION` y se copia
  en `DEFAULT_PROMPTS`.
- `compose.render()`: sin la etiqueta «Parte A:» cuando el ciclo tiene una sola pregunta.
- Sigue habiendo una sola repregunta por ciclo. La puntuación y la retroalimentación ya trabajan
  por parte.

### 4.3 · Banco agotado (red de seguridad)

Con el bloqueo al inicio (4.2), esto solo pasa si el profesor desactiva preguntas a mitad de un
intento. Decisión (usuario): el intento pasa a la encuesta **sin penalizar**.

`assessment-attempt.service.ts`:

- Helper `eligibleForOutcome(attempt, cycleIndex)`: la misma consulta que `next-cycle` (`active`,
  `rubricId` no nulo, `id notIn used`, resultado del ciclo). Se extrae a una función compartida
  para que no diverjan.
- `continue()` (~l. 666): si el siguiente ciclo no tiene ninguna pregunta elegible, un
  `updateMany` condicional (`phase: awaiting_continue`, `currentCycle`) fija
  `totalCycles = currentCycle + 1` y llama a `toSurvey`. `attemptFraction` divide entre ese valor,
  así que los ciclos que nunca existieron no cuentan como 0. Log
  `[Assessment] bank exhausted attempt=… presented=…`.
- `retry()` en `retry_compose` (~l. 699): el mismo cierre con `totalCycles = currentCycle`. El
  ciclo que no se pudo componer no tiene fila, porque el agente reporta el ciclo antes de
  mostrarlo. Con `currentCycle === 0` no se cierra (un intento con 0 ciclos no tiene nota): se
  registra un error y el intento se queda donde está.
- `is_last` del turno `answer` (~l. 841): también es verdadero si el siguiente ciclo no tendría
  preguntas, para que Pablo haga el cierre en vez de anunciar otro tema.
- Barrido: los intentos en `retry_compose` con `currentCycle > 0` y sin preguntas para su ciclo
  reciben el mismo cierre.
- Agente: sin cambios. `no_candidates` sigue reportándose y `/retry` aplica el cierre.

### 4.4 · «Escribiendo…» sin fin, fechas y vista previa

Decisiones (usuario): el error se muestra dentro del chat y vuelve el botón; antes de «Comenzar»
se avisa de las fechas; la vista previa ignora las fechas.

Plugin (`amd/src/attempt.js`):

- `fail()` siempre llama a `removeTyping()`. Si no hay intento cargado (`this.state === null`), el
  error va a `chat-status`, dentro del chat. Hoy va a `status`, que queda tapado por el chat
  (`position: fixed` y pantalla completa). Además vuelve el botón de inicio (`renderStart()` o
  `renderPreviewStart()`).
- Strings: `error_NOT_ENOUGH_QUESTIONS` («La actividad aún no está lista. Avísale a tu profesor.»;
  en la vista previa, lo que falta según `missing`) y uno genérico para errores de red.
- Fechas: `view.php` ya pasa `timeopen`/`timeclose`. Hay que pasar también las fechas formateadas
  con `userdate()`, en la zona horaria del usuario. En `renderStart()` (estudiante): antes de abrir,
  «La actividad abre el {fecha}.»; después de cerrar, «La actividad cerró el {fecha}.»; en ambos
  casos sin botón. La vista previa conserva siempre su botón.

Backend:

- `start()` (~l. 228): `NOT_OPEN` y `CLOSED` solo si `!caller.preview`.
- `sweep()` (~l. 1262, cierre por fecha): añadir `preview: false` a la consulta. Si no, una vista
  previa hecha después del cierre pasaría a la encuesta al minuto siguiente.
- Cada rechazo de `start()` deja `[Assessment] start refused code=… cmid=… user=… preview=…`.

### 4.5 · Datos de producción

- **`d2012350`**: se borra (decisión del usuario), **pidiendo confirmación explícita antes de
  ejecutarlo** y antes de desplegar. Si no se borra, el barrido de 4.3 lo cerraría y publicaría
  una nota en el libro de calificaciones del profesor. El borrado es
  `DELETE FROM assessment_attempts WHERE id = 'd2012350-23e8-45ac-a4b6-aebc4a80ff29'` por el
  túnel. Las tablas hijas se borran en cascada y la conversación se conserva. El
  `gradeRangeLockedAt` de 86226 sigue bloqueado.
- **`ec7cdd5d`** (86235, 27-oct): comprobar si es vista previa. Si es un intento real del
  profesor, decidir lo mismo después de la prueba.

### 4.6 · Interfaz del intento

Basado en la captura de `ec7cdd5d` (27-oct). Las decisiones de este apartado son propuestas y el
usuario todavía puede cambiarlas.

| # | Problema | Cambio | Dónde |
|---|---|---|---|
| 1 | El calentamiento se ofrece como lo normal, pero el botón principal se lo salta (se pulsó `/begin`) | Con calentamiento disponible y sin hacer (`warmup.count = 0`): botón principal «Hacer el calentamiento (sin nota)» y secundario «Saltarlo y empezar». Una vez hecho, el principal pasa a ser «Empezar con las preguntas». Los botones aparecen bajo el mensaje de apertura, dentro del chat, y no al pie | `attempt.js::renderActive` (`opening`), strings |
| 2 | El mensaje de apertura no dice qué pulsar | `opening_warmup`: «Antes, si quieres, hacemos una pregunta de prueba sin nota para probar el micrófono y el formato. Elige abajo: puedes saltarla.» | agente `texts.py` |
| 3 | La pregunta sale dos veces: en el chat y entera en el panel fijo «Pregunta abierta», que ocupa media pantalla y deja el chat en una franja | El panel aparece plegado en una línea («Pregunta abierta · Ver»). Al desplegarlo, `max-height: 35vh` con desplazamiento. Se oculta mientras la burbuja de la pregunta se ve en el chat (`IntersectionObserver`) | `attempt.js::markOpenQuestion`, `styles.css` |
| 4 | Las partes forman un solo bloque de texto | Cada parte en su propio párrafo con la etiqueta en negrita (líneas que empiezan por `Parte X:` o `Part X:` en los mensajes de pregunta). Con una sola pregunta no hay etiqueta (4.2) | `attempt.js::bubble`, agente `compose.render` |
| 5 | «Hablemos de Reconoce los mitos modernos… bíblica..»: el nombre del resultado es un objetivo con verbo y punto final, metido en una plantilla que espera un sustantivo | `compose.topic_name()` quita la puntuación final. Las plantillas pasan a formato rótulo, que vale tanto para un verbo como para un sustantivo: `progress_only` «Tema: {name}.», `progress_first` «Primer tema de {count}: {name}.», `opening_topic` «Vamos a trabajar sobre esto: {topics}.» | agente `compose.progress_line`, `texts.py` |
| 6 | «Hola, JORGE EDUARDO» | Si el nombre llega todo en mayúsculas, se pasa a mayúscula inicial («Jorge Eduardo») | agente `nodes.py` (~l. 437) |
| 7 | «Escribiendo…» sin fin y error invisible | Ver 4.4 | plugin |

---

## Orden de despliegue

⚠️ El backend valida con `forbidNonWhitelisted: true` (`src/main.ts:87`): si el agente envía
`verdicts` o `perQuestionFeedback`, o el plugin envía `graderInfo`, a un backend sin actualizar,
recibirá `400 Bad Request`. Lo mismo pasa con `questionsPerCycle` del plugin (4.2). El orden
estricto es:

0. **Datos (4.5)**: borrar `d2012350`, con confirmación, antes de desplegar el backend.
1. **Migración de base de datos** (`migrate.sh yorchio`), incluidas las columnas
   `questionsPerCycle` (4.2).
2. **Backend** (`yorchio-backend`).
3. **Agente** (`yorchio-agent`).
4. **Plugin Moodle 0.3.3** (`yorchio-moodle`); su `db/upgrade.php` encola `sync_setup` para poblar
   `graderInfo` en las actividades existentes.

---

## Verificación

- **Agente** (`uv run pytest tests/test_pablo_jobs.py tests/test_pablo_graph.py -v`):
  - `scoring`:
    - `CycleScore` exige veredicto para cada pregunta con `grader_info` antes de `components` (si
      falta, reintenta y falla con `scoring_failed` tras agotar reintentos);
    - pregunta sin `grader_info` recibe `no_reference` en código sin depender del modelo;
    - ciclo sin respuesta emite `incorrect` (`Sin respuesta.`) en preguntas con referencia y
      `no_reference` en las demás, sin llamar al LLM;
    - `grader_info` viaja cercado con `<<<REFERENCIA … REFERENCIA>>>` y no puede romper su cerca;
    - `reference_rule("score", lang)` se anexa aunque el tenant tenga un prompt personalizado.
  - `feedback`:
    - `perQuestionFeedback` cruza el comentario del modelo con el `verdict` y `part` deterministas
      y los renderiza bajo «Por pregunta» / «By question»;
    - guardia `copied_span`: si el modelo copia ≥ 6 palabras seguidas de `grader_info`, reintenta
      una vez y, si persiste, regenera sin referencias (`include_references=False`) registrando
      `pablo_feedback_reference_leak`.
  - `followup`, `compose` y `echo`:
    - `compose.candidates_block` y `echo` nunca reciben `grader_info`;
    - `followup.plan_for` incluye `<<<REFERENCIA … REFERENCIA>>>` y `followup.validated` rechaza
      una repregunta que filtre una palabra de `grader_info` ausente en la pregunta vista y en la
      respuesta del estudiante.
- **Backend** (`assessment-pure.spec.ts` y `assessment.flow.spec.ts`):
  - `setup` persiste y actualiza `graderInfo`;
  - `score` persiste `verdicts` y `feedback` acepta `perQuestionFeedback`;
  - `evidence` expone `graderInfo`, `verdict` y `reason` al profesor;
  - `view()`/`mine()` del estudiante incluyen `feedback.perQuestionFeedback` pero jamás
    `graderInfo` ni el `reason` del profesor;
  - retrocompatibilidad con un plugin que no envíe `graderInfo`;
  - preguntas por ciclo (4.2): `cyclesForOutcome` y `bankShortfall` con 1 y con 3 resultados,
    y con `M < n`. `start()` responde `409 NOT_ENOUGH_QUESTIONS` con `missing`, también en la
    vista previa. Con `k = 1`, 3 preguntas y 3 ciclos, el intento llega a la encuesta.
    `next-cycle` envía `questions_per_cycle` y ya no mezcla resultados. Un plugin que no envía
    `questionsPerCycle` conserva el valor guardado;
  - banco agotado (4.3): desactivar preguntas a mitad del intento. El turno `answer` lleva
    `is_last: true`, `continue` lleva a la encuesta con `totalCycles` recortado y la nota no
    penaliza el ciclo que no existió. Un intento en `retry_compose` sale con `/retry` y con el
    barrido. Con `currentCycle = 0` no se cierra;
  - fechas y vista previa (4.4): la vista previa empieza antes de abrir y después de cerrar, y el
    barrido no la cierra. Un estudiante recibe `NOT_OPEN` / `CLOSED`, y cada rechazo deja
    `start refused`.
- **Agente**, además: `validated_parts` con `k` = 1, 2 y 3 (recorta, completa y usa menos
  candidatas si no hay más); sin `questions_per_cycle` se queda en 2–3; `count_rule` se añade
  aunque el tenant tenga un prompt propio; `render` sin «Parte A» con una sola parte;
  `topic_name` quita la puntuación final; un nombre en mayúsculas sale con mayúscula inicial.
- **Plugin** (`tests/setup_test.php`):
  - `snapshot()` incluye `graderInfo` en preguntas `essay` y lo omite en otros tipos; `reference`
    booleano en `card()`;
  - `readiness()` con `questionspercycle`: `'enough'` por resultado y lo que falta detallado;
  - a mano (el JS no tiene pruebas): un 403 al empezar deja el error en el chat y el botón,
    nunca «Escribiendo…»; el aviso de fechas antes de abrir y después de cerrar; el
    calentamiento como botón principal; el panel «Pregunta abierta» plegado.
- **Producción, tras desplegar:** 86226 y 86235 quedan con `questionsPerCycle = 1` y como
  «listas»; un intento nuevo de 86235 recorre los 3 ciclos con una pregunta cada uno.
- **Prueba local de punta a punta**:
  1. Configurar una actividad con una pregunta Ensayo con «Información para evaluadores» y otra
     sin ella.
  2. Completar un intento con una respuesta conceptualmente correcta y otra con un error conceptual
     frente a la referencia.
  3. Verificar que la repregunta en vivo apunta al vacío conceptual sin filtrar palabras de la
     referencia, que la nota refleja la exactitud frente a la referencia, que el estudiante ve el
     desglose por pregunta (veredicto + orientación sin la respuesta esperada) y que el profesor ve
     la referencia y el veredicto con su motivo en la revisión.
