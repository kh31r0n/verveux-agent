# Pablo después de su primera evaluación real: plan de mejora

> Revisión de producción del 2026-11-07 (solo lectura, túnel IAP con el rol `app`).
> Tenant **Unidad Educativa Casa Roca Bogotá** (`a3ea6df8-…`), actividad **«QUIZ 20% MITOS DE LA
> MODERNIDAD»** (cmid 86235, `ab94d22d-…`): 23 intentos reales de 21 estudiantes del 29-oct al 3-nov,
> 64 ciclos, 59 repreguntas, 23 retroalimentaciones. Las vistas previas del profesor quedan fuera.
> Agente `git-0ac1b25` (`yorchio-agent-00030-4mr`), backend `git-eee2323`, plugin 0.3.3.

## Contexto

La actividad tiene 1 resultado de aprendizaje, 3 preguntas Ensayo (Descartes, Maquiavelo, Marx), cada
una con «Información para evaluadores», 1 pregunta por ciclo y 3 ciclos. Los tiempos son 15 min por
ciclo, 80 % para la repregunta y 60 min por intento. La escala va de 10 a 50 (aprueba con 30), la
modalidad es texto o audio y hay calentamiento.

Lo que funcionó: los 21 intentos llegaron a la encuesta y a la nota, y la experiencia se valoró con un
promedio de 4,6/5 en 20 encuestas. Varios comentarios valoran que la evaluación «toma tus propias
respuestas y las utiliza como nuevas preguntas». No hubo fallos de composición visibles, ni
calificaciones fallidas (2 reintentos internos se recuperaron), ni filtraciones de la referencia en la
retroalimentación.

## Hallazgos (con evidencia)

### A · Fiabilidad: se perdieron respuestas

1. **El audio nunca ha funcionado en producción.** Se emitieron 97 URLs de subida (`audio-upload` → 200)
   y no hay ni un objeto de audio en `gs://vervux-prod-yorchio-uploads`. Los 61 ciclos respondidos son de
   texto, y el agente no registró ni una transcripción en 14 días. Causa: el CORS del bucket solo admite
   `https://one.vervux.com`, así que el `PUT` firmado desde `aulas.unidadeducativa.org` lo bloquea el
   navegador. El calentamiento, que existe precisamente para probar el micrófono, tampoco pudo subir audio.
2. **Respuestas escritas descartadas por llegar tarde.**
   - `c470017f` (usuario Moodle 5406): el primer envío de la pregunta 2 llegó 8 s después del plazo
     (cierre más 5 s de gracia). Recibió `409 TIME_UP` en ese envío y en 5 reintentos (21:00:36 →
     21:01:58). El texto nunca llegó al servidor, quedó calificado «No respondiste» con 0 puntos, y el
     estudiante escribió en la encuesta: «me siento estafado».
   - `a588cea6` (650): grabó audio, pidió la URL a las 04:40:01 y el envío llegó 0,9 s fuera de la
     gracia → 409.
   - `af0c58ec` (4184): grabó 60 s de audio, la subida falló (punto 1) y el ciclo cerró por tiempo
     vacío, con 0 puntos. Tiene abierto el caso «una respuesta la grabé pero no se montó».
   - Probable causa del retraso en el texto: el temporizador del navegador (`setInterval`) se ralentiza
     cuando la pestaña queda en segundo plano. Hoy el servidor no tiene nada del texto hasta el envío.
3. **Intentos sin ninguna respuesta calificados con la nota mínima.** El barrido del cierre los mandó
   a la encuesta abandonada y luego a calificación, con nota 10/50:
   - `0c258a0d` (5465) empezó 21 min antes del cierre y se le cortó la primera pregunta a los 3 min.
   - `aeb9401c` (4539, segundo intento) nunca pasó de la apertura. Con «mejor nota» no le afecta (vale
     su 35), pero con «promedio» le restaría.
4. **La rúbrica tiene 13 versiones idénticas.** `assessment-config.service.ts` compara
   `JSON.stringify(current.components)` con lo recibido. Postgres (`jsonb`) reordena las claves
   (`id, name, criteria, maxPoints`) y el plugin las manda como `id, name, maxPoints, criteria`, así que
   cada sincronización crea una versión nueva. Efecto: los ejemplos de calibración y las aclaraciones
   de rúbrica, que van por `rubric_version_id`, dejan de aplicar en cuanto el profesor vuelve a guardar.

### B · Calificación: el veredicto y los puntos se contradicen

5. **«Parcialmente correcta» con la nota máxima.** De 32 veredictos `partial`, 14 tienen 50/50.
   `correct` promedia 49,8, `partial` 39,5 e `incorrect` 3,0.

   | Veredicto | n | 50 | 40 | 35 | 30 | 15 | 0 |
   |---|---|---|---|---|---|---|---|
   | correct | 27 | 26 | — | — | — | — | — (1 con 45) |
   | partial | 32 | **14** | 4 | 6 | 5 | 3 | — |
   | incorrect | 5 | — | — | — | — | 1 | 4 |

   La regla «el veredicto es verdad base» del plan anterior solo está en el prompt; nada en código la
   hace cumplir.
6. **La referencia exige contenido que el enunciado no pide.** La pregunta de Descartes salió
   `partial` en los 18 intentos que la respondieron (ningún `correct`), casi siempre con el mismo
   motivo: «omite el contraste con la humildad de Newton y la relación con Santiago 4:4». La
   referencia dice «Debe mencionar…», pero el enunciado no lo menciona (viene de la conferencia). En
   Maquiavelo y Marx, los `partial` citan puntos marcados como «Puede…» (la antropología de
   Maquiavelo, Ro 3:23, Silva-Silva) o temas ausentes del enunciado («la conexión con Rousseau»,
   «capítulo III»). Hoy el modelo no distingue entre lo obligatorio, lo opcional y los errores
   frecuentes.
7. **La rúbrica es gruesa.** Tiene un solo componente de 50 puntos con niveles 0/3/6/10. El modelo los
   traduce ×5 (0/15/30/50) e interpola (35, 40, 45). Es una sola nota por pregunta, sin desglose.

### C · Conversación

8. **La repregunta se agota por tiempo.** 21 de 83 llamadas superaron los 8 s
   (`pablo_followup_failed` con error vacío, es decir `asyncio.TimeoutError`). Usa el modelo del
   tenant `gemini-3.5-flash`, de media 1.544 tokens de entrada y 127 de salida. 12 de las 59
   repreguntas fueron el respaldo en código.
9. **Repreguntas con fórmula repetida.** De las 47 que escribió el modelo, 21 empiezan con
   «¿Cómo se relaciona…» y 14 más con «¿De qué manera / Cómo se contrasta / se diferencia /
   se contrapone…». Además, a menudo repiten la consigna original («…con lo que señala Romanos 3:8»)
   en lugar de avanzar sobre lo que dijo el estudiante.
10. **Texto corrupto del modelo.** Se vieron «²Qué es exactamente…», «²Cómo se relaciona…» y
    «!Hola! !Qué gusto saludarte de nuevo! ¹Cómo se relaciona…», con signos de apertura mal
    codificados y un saludo dentro de la repregunta.
11. **El respaldo con cita queda mal.** Corta en la palabra 14 a mitad de frase («…donde plantea
    pienso luego»), copia mayúsculas sostenidas («EL MAL NACE EL CORAZON…») y puede atribuirle al
    estudiante una idea que solo estaba describiendo: «Dijiste «Dios es una construcción de la mente».
    ¿Qué cambiaría si no fuera así?», cuando estaba explicando a Marx. Además «¿Por qué lo ves así?»
    presupone una postura.
12. **El guardia léxico rechaza palabras comunes.** Hubo 16 rechazos por `leak:` con palabras como
    «entre», «forma», «origen», «propuesta», «tierra», «capacidad», «pensar» o «bíblicos». Algunas
    están en el enunciado original del banco pero no en la versión compuesta que se compara. Otras sí
    eran fugas reales («opera», «sometida», «revelación», tomadas de la referencia).
13. **Encabezado repetitivo con un solo tema.** Los ciclos 2 y 3 dicen «Volvemos al tema «Reconoce los
    mitos modernos y realiza un contraste con la visión bíblica» con otra pregunta.».

### D · Operación (no es código)

- 7 casos de revisión siguen abiertos y sin decisión: 4 de muestreo, 2 por pegado bloqueado y
  1 pedido por el estudiante.
- 9 de 60 respuestas se enviaron solas al vencer los 15 min. Tres comentarios de la encuesta hablan de
  presión de tiempo.

## Decisiones (acordadas con el usuario)

| Tema | Decisión |
|---|---|
| Veredicto vs. puntos | La referencia se clasifica en «Debe», «Puede» y «Error frecuente», y solo un «Debe» omitido o un error frecuente cometido baja el veredicto. En código, `partial` no puede sacar el máximo e `incorrect` no pasa de la mitad. Se avisa al profesor cuando un «Debe» no está en el enunciado. |
| Respuestas perdidas | Autoguardado de borrador; al vencer el plazo cuenta el último borrador. Un audio recibido antes del plazo cuenta como enviado a tiempo. |
| Audio | Se sube por el backend, que ya tiene CORS para Moodle; nada de `PUT` directo al bucket. |
| Repregunta | Hasta 20 s con el modelo del tenant, dentro del límite del turno. |
| Intentos sin respuesta | Quedan «abandonados», sin nota ni libro. Antes de «Comenzar» se avisa si la actividad cierra antes de que alcance un ciclo. |
| Estudiantes afectados | Lista con evidencia para el profesor y una acción nueva «Reabrir esta pregunta». |
| Textos | Repreguntas sin fórmula repetida, limpieza de la repregunta y del respaldo, y encabezado propio cuando hay un solo tema. **La retroalimentación no cambia** (no se pidió). |
| Intentos ya cerrados | **No se recalifican** con las reglas nuevas: el tope por veredicto bajaría notas ya publicadas. El profesor decide caso a caso con la revisión. |

---

## Ramas

| Repo | Rama nueva | Desde |
|---|---|---|
| `yorchio-backend` | `feat/assessment-production-fixes` | `feat/assessment-grader-info` (producción, `eee2323` + espejo del prompt) |
| `yorchio-agent` | `feat/pablo-production-fixes` | `feat/pablo-grader-info` (`0ac1b25`, producción) |
| `yorchio-moodle` | `feat/assessment-production-fixes` | `feat/assessment-grader-info` (0.3.3) |

⚠️ Los checkouts están ahora en `feat/caleb-fase-8` (trabajo de otra línea): crear las ramas en
worktrees o confirmar antes de cambiar de rama.

---

## Fase 1 · Fiabilidad (P0)

### 1.1 Audio por el backend (hallazgo 1)

- **Backend:**
  - Endpoint nuevo `POST /moodle/assessments/:cmid/attempts/:id/audio?practice=0|1`, con cuerpo binario
    `audio/(webm|ogg|mp4|mpeg|wav)` de hasta 4 MB (`express.raw` solo en esa ruta). Usa la misma
    validación de modalidad y de fase que `audioUploadUrl`.
  - Guarda en `ObjectStorage.put(key, body, contentType)` con la misma clave de hoy (`audioPrefix`) y
    devuelve `{audioKey}`. Si el ciclo está abierto, además lo registra como borrador de audio (1.2).
  - `audio-upload` (URL firmada) queda solo para clientes viejos.
  - CORS: la ruta ya está bajo `/api/v1/moodle/assessments`, que tiene su CORS propio en `main.ts`.
- **Plugin (`amd/src/attempt.js`):**
  - Sube con `fetch(apiBase…/audio, {body: blob})` en vez del `PUT` firmado, apenas termina la grabación
    y antes de «Enviar». El calentamiento usa la misma ruta.
  - Si la subida falla, se muestra el error dentro del chat y se ofrece reintentar. Ya no se descarta
    en silencio.
- **Prueba:** el recorrido de navegador con audio (`dev/assessment-browser.mjs`, micrófono falso de
  Chromium) contra un origen distinto del CRM. Hoy el test pasa porque en local no hay CORS de bucket.

### 1.2 Autoguardado de borrador (hallazgo 2)

- **Migración aditiva** en `assessment_cycles`: `draftText`, `draftAt`, `draftAudioKey`,
  `draftAudioAt`, y lo mismo con prefijo `followup*`.
- **Backend:**
  - Endpoint `PUT /moodle/assessments/:cmid/attempts/:id/draft {cycleIndex, part, text}`. Solo se
    acepta con la fase abierta y `now ≤ deadline + gracia`; fuera de eso responde 204 sin guardar.
    Throttle propio (`@Throttle({ long: … })` con nombre configurado; ver `CLAUDE.md`).
  - `applyTimeouts`: si el ciclo vence con borrador (texto o audio recibido antes del plazo), lo
    convierte en la respuesta (`responseText = draftText`, `audioKey = draftAudioKey`,
    `timedOut: true`) y corre `runAnswerTurn` (turno `answer`, con repregunta si corresponde) en lugar
    del turno `timeout`.
  - `answer()` tarde: si el ciclo ya se cerró con su borrador, devuelve el estado (idempotente) en vez
    de `409 TIME_UP`. Sin borrador, el 409 se mantiene.
  - La integridad no cambia: la telemetría solo llega con el envío. Un borrador no la trae, y eso se
    marca (`telemetry: null`, alerta ninguna).
- **Plugin:**
  - Guarda el borrador cada 10 s si el texto cambió, y en `visibilitychange: hidden` y `pagehide` con
    `fetch(…, {keepalive: true})`.
  - Ante un 409, recarga el estado y muestra «Se acabó el tiempo; guardamos lo que alcanzaste a
    escribir» cuando el servidor usó el borrador. Hoy deja reintentar en vano.
- **Prueba de flujo** (`assessment.flow.spec.ts`): borrador a mitad de ciclo, envío 8 s tarde, y
  comprobar que la respuesta queda con el texto del borrador, `timedOut`, el turno `answer` con
  `timed_out: true` y sin 409.

### 1.3 Intentos sin respuesta y aviso de cierre (hallazgo 3)

- **Backend:**
  - Antes de `startScoring`, si ningún ciclo tiene respuesta con contenido (texto, transcripción,
    audio o repregunta), el intento pasa a `status: "abandoned"`, sin `grade`, sin calificación ni
    retroalimentación, y la conversación se cierra. Lo aplican el barrido de cierre, `abandonSurvey`
    y `survey`.
  - Queda fuera de `pendingGrades`, `gradebookGrade` y `buildInputs`. Sigue contando como intento
    usado para `maxAttempts`.
  - `mine()`/`view()` muestran el estado «Intento sin respuestas».
- **Datos:** `0c258a0d` y `aeb9401c` pasan a `abandoned` con `gradeVersion + 1`, para que el próximo
  pull reescriba el libro. Se ejecuta por túnel con el rol `migrator` y **con confirmación**.
  - ⚠️ Antes, verificar que `gradebook::pull` del plugin borra la nota cuando el CRM devuelve
    `grade: null`. Si no, añadirlo.
- **Plugin (`renderStart`):** si `timeclose − ahora < cycleduration` (o el tiempo del intento, el que
  sea menor), muestra «La actividad cierra a las {hora}: tendrás unos {n} minutos» sobre el botón,
  sin bloquearlo. `view.php` ya pasa `timeclose`; falta pasar `cycleduration`.

### 1.4 Versiones de rúbrica idénticas (hallazgo 4)

- `assessment-config.service.ts`: comparar con un `stableStringify` de claves ordenadas, o comparar
  campo a campo `id/name/maxPoints/criteria`.
- Prueba: sincronizar dos veces la misma rúbrica deja una sola versión.
- Sin limpieza de las 13 versiones existentes: son inocuas.

---

## Fase 2 · Calificación (P1)

### 2.1 Debe / Puede / Error frecuente y tope por veredicto (hallazgos 5 y 6)

- **`common.reference_rule("score")`**, reescrita:
  - Antes de juzgar, el modelo clasifica cada punto de la referencia como:
    - `requerido`: «Debe…», «Obligatorio…», o contenido central que el enunciado pide;
    - `opcional`: «Puede…», «Opcional…», o contenido que el enunciado no pide;
    - `error`: «Error frecuente…».
  - Veredicto: `correct` si cubre todo lo requerido y no comete errores; `partial` si omite algo
    requerido o lo cubre a medias; `incorrect` si contradice lo esencial o comete un error frecuente.
  - Un punto opcional **nunca** baja el veredicto. Se mantienen las reglas de cerca de la referencia
    y de no obedecer puntajes escritos dentro.
- **Esquema `QuestionVerdict`:** se añaden `missing_required: list[str]` y `errors_found: list[str]`
  (frases cortas, para el profesor). `reason` se mantiene.
- **Tope en código** (`scoring.components_from`, por `question_snapshot_id`):
  - `partial`: puntos ≤ `PABLO_PARTIAL_CAP` × máximo (0,8 por defecto).
  - `incorrect`: puntos ≤ `PABLO_INCORRECT_CAP` × máximo (0,5).
  - Se marca `clamped: true`, se añade «(ajustado al veredicto)» al `rationale`, se registra el log
    `pablo_points_capped_by_verdict` y el contador `pablo_points_capped_total{verdict}`.
- **Backend:** `QuestionVerdictDto` acepta `missingRequired?` y `errorsFound?` (arreglos de cadenas,
  ≤ 10 × 300) y se guardan en columnas `missingRequired TEXT[]` y `errorsFound TEXT[]` de
  `assessment_question_verdicts`. `evidence` los expone.
- **Plugin:** la revisión del profesor los muestra bajo el veredicto («Faltó: …» / «Error: …»).
- **Retroalimentación:** `reference_rule("feedback")` añade «no presentes como faltante un punto
  opcional». No es un cambio de formato, es coherencia con el veredicto.
- **Prompts:** `SCORE_PROMPT` y `FEEDBACK_PROMPT` → nueva versión, con espejo en `DEFAULT_PROMPTS`.
- **Pruebas:** que una referencia con solo «Puede» no baje de `correct`; el tope en `partial` e
  `incorrect` con varias preguntas por ciclo; que el profesor vea `missingRequired`.

### 2.2 Aviso al profesor: «Debe» fuera del enunciado (hallazgo 6)

- **Agente:** `POST /assessment/reference-check` (síncrono, servicio). Por pregunta, recibe
  `{text, grader_info}` y hace una llamada estructurada con el modelo de fondo. Devuelve
  `required_not_in_statement: list[str]` y `optional_items: list[str]`. Usa la cerca `REFERENCIA` y
  aplica en código que cada elemento devuelto sea una subcadena de la referencia.
- **Backend:**
  - En `config.sync`, para cada pregunta cuyo `sha256(text + graderInfo)` cambió, la revisa en
    segundo plano y guarda `referenceCheck JSON` y `referenceCheckSha` en `assessment_questions`
    (migración aditiva).
  - El resultado se expone en la respuesta del sync y en un `GET :cmid/questions/checks`.
- **Plugin:**
  - En el chip de la pregunta, un aviso «La referencia exige «humildad de Newton; Stg 4:4», que el
    enunciado no pide: Pablo lo tratará como requerido».
  - `readiness()` cuenta esas preguntas, solo como aviso (no bloquea).
- Costo: una llamada barata por pregunta editada.

---

## Fase 3 · Conversación (P2)

### 3.1 Repregunta dentro del tiempo (hallazgo 8)

- `PABLO_TIMEOUT_FOLLOWUP_SECONDS` sube a 20.
- Hay un presupuesto por turno: `pablo_receive_answer_node` anota cuándo empezó, y `plan_for` recibe
  `deadline = inicio + PABLO_FOLLOWUP_BUDGET_SECONDS` (40 s). Así el turno queda por debajo de
  `AGENT_TOTAL_TIMEOUT_MS = 60 s` del backend aunque la transcripción tarde.
  - El primer intento usa `min(20, restante)`.
  - El reintento solo se hace si quedan al menos 8 s, con `restante` como límite.
- Prueba: con un proveedor falso lento, el turno no supera el presupuesto y sale el respaldo.

### 3.2 Sin fórmula repetida (hallazgo 9)

- `state["pablo"]["followups"]`: las repreguntas ya hechas en el intento (texto, hasta 3).
- `plan_for` las añade al mensaje: «Repreguntas que ya hiciste en este intento (no repitas su forma
  ni su apertura)».
- `MOVES_RULE` añade dos reglas: no reformular la consigna original, y avanzar sobre algo que el
  estudiante dijo y que la consigna no le daba hecho.
- `validated()` rechaza con `repeated_opening` cuando:
  - las tres primeras palabras plegadas coinciden con las de una repregunta anterior del intento, o
  - empieza con «¿cómo se relaciona» y esa apertura ya se usó una vez en el intento.

  `refusal_note` lo explica en el reintento.
- `FOLLOWUP_PROMPT` → nueva versión, con espejo en el backend.

### 3.3 Limpieza de la repregunta y del respaldo (hallazgos 10 y 11)

- `followup.tidy(question)`, aplicada antes de validar:
  - un `¹`, `²`, `³` o `?` inicial pasa a `¿`, y un `!` inicial a `¡`;
  - se quitan frases iniciales de saludo o elogio («¡Hola!», «¡Qué gusto…!», «¡Excelente!», «Muy
    bien…») hasta la primera pregunta;
  - si termina en «?» y no tiene «¿» (español), se antepone.
- Respaldo con cita (`answer_fragment`):
  - toma la primera oración completa si tiene 3–20 palabras;
  - si es más larga, corta en la última coma o conjunción dentro de las 20 palabras;
  - si no hay corte limpio, no cita y usa la genérica;
  - pasa las mayúsculas sostenidas a mayúscula inicial. La comparación literal ya pliega
    mayúsculas, así que la cita sigue validándose.
- Plantillas neutras, que no presuponen la postura del estudiante:
  - «Escribiste «{quote}». ¿Puedes explicar un poco más esa idea?»
  - «Sobre esta parte de tu respuesta: «{quote}». ¿Qué pasaje o ejemplo la sostiene?»
  - «Retomo lo que escribiste: «{quote}». ¿Qué consecuencia tiene esa idea?»

### 3.4 Guardia léxico afinado (hallazgo 12)

- Lo permitido pasa a ser la pregunta compuesta, la respuesta **y los enunciados originales del banco**
  de las preguntas del ciclo. `cycle["question_texts"]` se guarda en el estado al componer, solo para
  esto.
- Comparación por raíz plegada (los primeros 6 caracteres), así «bíblicos» ≈ «bíblica».
- Se amplía `_GENERIC` con palabras frecuentes del español de 5 o más letras («entre», «forma»,
  «sobre», «origen», «propuesta», «tierra», «parte», «desde», «cuando», «donde», «tiene», «puede»,
  «hacer»…), como lista fija en código.
- Siguen rechazándose las fugas reales («opera», «sometida», «revelación» tomadas de la referencia).
- Prueba con los 16 rechazos de producción: las 8 palabras comunes pasan y las de la referencia no.

### 3.5 Encabezado con un solo tema (hallazgo 13)

- `compose.progress_line`, cuando `outcome_count == 1`:
  - «Primera pregunta de tres.», «Segunda pregunta de tres.» y la última termina con «Es la última
    pregunta.»;
  - sin repetir el nombre del resultado; el tema ya se dijo en la apertura.
- Con 4 o más preguntas en total, se usan cifras («Pregunta 4 de 6.»).

---

## Fase 4 · Reabrir una pregunta (estudiantes afectados)

Para que el profesor pueda devolverle a un estudiante una pregunta que perdió por un fallo del sistema.

- **Backend:**
  - Endpoint `POST /moodle/assessments/:cmid/attempts/:id/cycles/:n/reopen {rationale}` con capacidad
    `review`, solo para un intento `scored`.
  - Queda una `AssessmentReviewDecision` con `action: "reopen"` (columna nueva o valor nuevo).
  - El intento pasa a `status: active`, `phase: "composing"`, `currentCycle: n` y `reopenCycle: n`
    (columna nueva). El ciclo `n` limpia su respuesta y su repregunta pero conserva
    `compositePrompt` y `components`.
  - Turno nuevo `reopen`: el backend envía `cycle_index`, `composite_prompt` y `component_map` desde
    la fila.
  - Al grabarse la respuesta, `response_recorded` con `is_last: true` lleva directo a calificar, sin
    encuesta (ya la hizo) y con `scoringVersion + 1`. Se reescriben la nota y la retroalimentación.
- **Agente:** el nodo `pablo_reopen` presenta el texto guardado tal cual, sin componer ni reportar el
  ciclo, con una cabecera fija «Tu profesor reabrió esta pregunta. Tienes el tiempo completo.». Emite
  `cycle_presented`. La repregunta funciona igual.
- **Plugin:**
  - En la revisión, un botón «Reabrir esta pregunta» por ciclo, con un motivo obligatorio.
  - Se envía un mensaje de Moodle al estudiante.
  - `view.php` muestra el intento como «en curso» con su chat.
- **Pruebas:** flujo completo, que el reloj solo corra en el ciclo reabierto y que la nota nueva
  reemplace a la anterior en el libro.

---

## Operación (sin código, para el profesor y el equipo)

1. **Estudiantes afectados (evidencia en este plan):**

   | Intento | Usuario Moodle | Qué pasó | Nota | Sugerencia |
   |---|---|---|---|---|
   | `c470017f` | 5406 | Pregunta 2 (Descartes): el texto no llegó (409 ×6) | 37 | Reabrir la pregunta 2 (Fase 4) o decidir la nota |
   | `a588cea6` | 650 | Pregunta 1 (Descartes): audio enviado 0,9 s tarde | 37 | Reabrir la pregunta 1 |
   | `af0c58ec` | 4184 | Pregunta 2 (Marx): audio no subido (CORS) | 27 | Reabrir la pregunta 2; tiene caso abierto |
   | `0c258a0d` | 5465 | El cierre cortó la primera pregunta a los 3 min | 10 | Pasa a «abandonado» (1.3) o se le da otro intento |
   | `aeb9401c` | 4539 | Segundo intento sin empezar | 10 | Pasa a «abandonado» (1.3); su nota en el libro sigue siendo 35 |

2. **Casos abiertos:** revisar los 7 de la cola (4 de muestreo, 2 por pegado bloqueado, 1 del
   estudiante).
3. **Referencia de Descartes:** decidir si «humildad de Newton / Stg 4:4» es obligatorio. Si lo es,
   añadirlo al enunciado; si no, escribirlo como «Puede…». Con la Fase 2 en marcha, un «Debe» que el
   enunciado no pide deja a todos en `partial` con tope de 40/50.
4. **Rúbrica:** proponer 2–3 componentes, por ejemplo «enunciado del mito», «fundamento bíblico» y
   «contraste», con puntos en la misma escala que sus niveles. Hoy hay un solo componente de 50
   puntos con niveles 0/3/6/10.
5. **Tiempo:** 9 de 60 respuestas se enviaron solas a los 15 min, y la repregunta tiene el 80 %.
   Considerar 20 min por pregunta para respuestas de reflexión.
6. **Audio:** hasta desplegar 1.1, poner la actividad en «solo texto» o avisar a los estudiantes de
   que el audio no funciona.

---

## Orden de despliegue

1. **Backend** con las migraciones de 1.2, 2.1, 2.2 y 4 (todas aditivas) → **agente** → **plugin
   0.3.4**.
2. 1.1 y 1.2 necesitan el plugin nuevo para surtir efecto. El backend nuevo sigue aceptando
   `audio-upload` y no exige borradores, así que el plugin viejo funciona igual que hoy.
3. Datos de 1.3 (`0c258a0d`, `aeb9401c`) **con confirmación explícita**, después de desplegar el
   backend.
4. Sin recalificar los intentos cerrados.

## Verificación tras desplegar (consultas para repetir esta revisión)

- **Audio:** objetos en el bucket con el prefijo de evaluaciones, y `modalityUsed in ('audio','both')`.
- **Envíos tarde:** `409 TIME_UP` en `…/answer` (logs HTTP) → debe tender a 0. Ciclos `timedOut` con
  borrador usado.
- **Coherencia:** ningún `partial` con el máximo y ningún `incorrect` por encima de la mitad, por
  pregunta.
- **Repregunta:** `fallback` en `pablo_followup_presented` (< 5 %), `pablo_followup_failed` (≈ 0) y
  aperturas distintas por intento.
- **Rúbrica:** una versión por cambio real.
- **Abandonados:** sin notas de 10 por intentos sin respuesta.
