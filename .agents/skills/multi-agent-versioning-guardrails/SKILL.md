---
name: multi-agent-versioning-guardrails
description: "INVOKE THIS SKILL when editing anything that touches the multi-agent versioning system: agent code names (helena/sofia/giulia/marco), agent assignments, conversation agentCodeName/agentVersion snapshots, the Python CODE_NAME_REGISTRY, the scoped thread ID format, the /chat/stream + /chat/resume request schemas, or the NestJS /channel-connections/:id/agent-assignments endpoints. Specific files: verveux-agent/src/graphs/registry.py, verveux-agent/src/main.py (lifespan + chat endpoints), verveux-agent/src/auth/service_auth.py (scoped_thread_id), verveux-agent/src/graphs/state.py, verveux-agent/src/observability.py, verveux-backend/prisma/schema.prisma (AgentCodeName, AgentAssignment, ChannelConnection.agentCodeName, Conversation.agentCodeName/agentAssignmentId/agentVersion), verveux-backend/src/agent-assignments/**, verveux-backend/src/conversations/conversations.service.ts (findOrCreateForContact), verveux-backend/src/channels/channel-connections.service.ts (create + update), verveux-backend/src/prisma/prisma.service.ts (conversation snapshot guard), and the frontend channels/switch-agent flow."
---

<overview>
The multi-agent versioning system replaced the legacy mutable `ChannelConnection.agentType` enum with permanent **agent code names** (`helena`, `sofia`, `giulia`, `marco`), explicit **agent assignment** rows (audit trail), and immutable **conversation snapshots**. Switching an agent on a connection cuts a new version-namespace boundary so in-flight conversations finish with the agent that started them and new conversations get a fresh checkpoint scope.

This skill encodes the eight load-bearing invariants and the patterns required to keep them holding. **Every invariant violation has caused a class of state-leak or routing bug before — none of these are stylistic preferences.**

Apply this skill before writing or reviewing any code in the file list above.
</overview>

---

## The eight load-bearing invariants

These are the non-negotiable rules. Each one corresponds to a class of bug that motivated the redesign. Never write code that violates one without making the violation explicit in a PR description.

1. **Conversation snapshots are immutable.** `Conversation.agentCodeName`, `Conversation.agentAssignmentId`, `Conversation.agentVersion` are set exactly once at creation (`findOrCreateForContact`) and never updated again. There is a Prisma `$extends` query interceptor in `verveux-backend/src/prisma/prisma.service.ts` that throws on `update`/`updateMany`/`upsert.update` if any of these fields are present. **Do not remove or bypass the guard.**

2. **One active assignment per connection.** `agent_assignments` has a partial unique index: `CREATE UNIQUE INDEX agent_assignments_one_active_per_connection ON agent_assignments(channelConnectionId) WHERE isActive = true`. Any code path that flips `isActive = true` on a row must first revoke the current one in the same transaction.

3. **Switchover is one DB transaction.** The full switch — validate code name → validate `tenant.availableAgents` → revoke current → insert new → update `ChannelConnection.currentAgentAssignmentId` + `agentCodeName` — happens inside `prisma.$transaction`. Do not split the steps. Do not touch `Conversation` rows during the switch.

4. **The Python service never silently routes to a default agent.** If `agent_code_name` is missing or unknown, the request returns `HTTP 400` with the unknown name and the list of registered code names. The legacy fallback path (`AGENT_TYPE_FALLBACK`) is gated behind a phase-1 flag and emits the `legacy_agent_type_fallback_total` canary metric.

5. **Startup validation against the backend.** `src/main.py` lifespan calls `fetch_active_code_names()` from `src/agents/backend_client.py`, which hits `GET /api/v1/internal/agent-versioning/active-code-names` on the NestJS backend (the source of truth for `agent_code_names`). It exits with `SystemExit` if the response lists active code names absent from `CODE_NAME_REGISTRY`. Registry-extra code names produce only a warning (they may be future-staged). Do not turn the fatal mismatch into a warning, and do not bypass the HTTP call by reading the table directly — the agent's own DB does not own this state.

6. **Thread IDs encode both `agentCodeName` and `agentVersion`.** `scoped_thread_id(tenant_id, user_sub, conversation_id, agent_code_name, agent_version)` returns five segments: `"{tenant_id}:{user_sub}:{conversation_id}:{agent_code_name}:v{agent_version}"`. Never construct a thread ID by string concatenation outside of this helper.

7. **Code names are permanent and single-builder.** Once a code name appears in `CODE_NAME_REGISTRY`, it is **frozen to one graph builder forever**. Renaming an existing code name, repointing it at a different builder, or deleting a row from `agent_code_names` that is referenced by any historical `AgentAssignment` is forbidden.

8. **Adding a new agent never edits existing builders.** A new code name is purely additive: a new graph file, a new entry in `CODE_NAME_REGISTRY`, a new row in `agent_code_names`. Do not touch shared state fields, existing graph builders, or routing logic for other agents.

---

## Cross-cutting rules

<db-schema>

### Prisma schema (`verveux-backend/prisma/schema.prisma`)

- `AgentCodeName` is **platform-level** (no `tenantId`). `codeName` is the primary key.
- `AgentAssignment` is tenant-scoped. The FK to `AgentCodeName.agentCodeName` uses `onDelete: Restrict` — never `Cascade`. Deleting an active code name must be impossible while history references it.
- `ChannelConnection.currentAgentAssignmentId` is `@unique` and FK-Restrict. This anchors the "exactly one current pointer" invariant.
- `Conversation` keeps `agentCodeName String?`, `agentAssignmentId String?`, `agentVersion Int @default(1)` — all three are written together at creation and never updated.

### Migrations

- Any change to the four models above (`AgentCodeName`, `AgentAssignment`, `ChannelConnection.agentCodeName`, `Conversation.agentCodeName/agentAssignmentId/agentVersion`) requires a new migration. Do not edit existing migration SQL files.
- Backfill migrations must be **idempotent**: every INSERT uses `ON CONFLICT DO NOTHING`, every UPDATE has a `WHERE` clause that becomes a no-op on second run.

</db-schema>

<nestjs-backend>

### Backend (`verveux-backend/src/`)

- **`ChannelConnectionsService.create`** must create the connection AND the initial `AgentAssignment` AND update the connection pointer in a single `prisma.$transaction`. The seed code name comes from `pickSeedCodeName(agentType)` — the first active code name for the requested type. Side effects (gateway provisioning, event emission) happen **after** the transaction commits, not inside.
- **`ChannelConnectionsService.update`** must reject any DTO with `agentType` set, with a 400 directing the caller to `POST /channel-connections/:id/agent-assignments`.
- **`AgentAssignmentsService.switchAgent`** is the only path that mutates `ChannelConnection.currentAgentAssignmentId` and `agentCodeName`. Conversation rows are **never** touched here — the `updateMany({ agentVersion: { increment: 1 } })` block that used to live in the legacy `update()` was the bug this whole system replaces. Do not bring it back.
- **`ConversationsService.findOrCreateForContact`** must read `connection.currentAgentAssignmentId` and `connection.agentCodeName`. If `currentAgentAssignmentId` is null, throw `InternalServerErrorException`. Compute `agentVersion` as `count(agent_assignments WHERE channelConnectionId = ? AND assignedAt <= activeAssignedAt)`.
- **`PrismaService.installConversationSnapshotGuard`** must remain installed via `Object.defineProperty(this, 'conversation', ...)` in `onModuleInit`. Removing or weakening the guard is the most dangerous change you can make in this codebase — verify the guard fires with a unit test if you touch this file.

</nestjs-backend>

<python-agent>

### Python agent (`verveux-agent/src/`)

- **`registry.py`** holds `CODE_NAME_REGISTRY: dict[str, GraphBuilder]`. Lookups go through `get_or_compile_graph(code_name, checkpointer)`, never directly. The compile path is guarded by `asyncio.Lock()` with a double-checked entry — do not rewrite this without preserving the lock semantics or the `graph_compile_duration` histogram observation.
- **Unknown code names raise `UnknownCodeNameError`**, never fall through to a default. The FastAPI handler turns this into HTTP 400.
- **`main.py` lifespan** validates backend↔registry before warming up, in that order. Both calls go through `src/agents/backend_client.py` (`fetch_active_code_names()` and `fetch_in_use_code_names()`), which hit `GET /api/v1/internal/agent-versioning/*` on the NestJS backend. Warm-up must be scoped to `fetch_in_use_code_names()` (i.e. only code names currently assigned somewhere). Do not warm up the full registry — that's wasted memory for staged code names not yet rolled out. Do not move the validation back into a direct DB query; the agent and backend may live in different Postgres databases.
- **`scoped_thread_id`** in `src/auth/service_auth.py` takes exactly five arguments. Adding a sixth argument or changing the separator silently invalidates every existing checkpoint. If the format must change, write a migration plan that includes thread-ID rewriting or accept that checkpoints will be lost.
- **`AgentState`** in `src/graphs/state.py` carries `agent_code_name` and `agent_version` for observability only. Nodes must not branch on these fields — routing is the graph's job, encoded by which graph was selected.
- **Observability counters** (`agent_requests_total`, `agent_node_invocations_total`, `agent_tool_errors_total`, `graph_compile_duration`) all carry the `agent_code_name` label. Adding new counters? Carry the label. Removing the label collapses all agents into one row in Grafana — never do this.

</python-agent>

<frontend>

### Frontend (`verveux-frontend/src/app/features/channels/`)

- The switch dialog (`dialogs/switch-agent-dialog.component.ts`) requires the admin to type the target agent's `displayName` to confirm. Do not remove this guard — it's the only thing standing between an accidental click and a state-namespace cutover that affects every new conversation.
- The selectable code names in the dropdown are filtered by `c.isActive && c.agentType ∈ tenant.availableAgents`. Both filters must remain — the first respects the platform kill switch, the second respects the tenant's enablement.
- The store's `switchAgent` must refetch the connection after a successful POST. Do not mutate local state in place — the backend computes the new `agentCodeName` and `currentAgentAssignmentId` and they must come from the server.

</frontend>

---

## Forbidden patterns (auto-fail in review)

<anti-patterns>

```ts
// ❌ NestJS: mutating snapshot fields after creation
await prisma.conversation.update({
  where: { id },
  data: { agentCodeName: 'sofia' },  // PrismaService guard throws — and rightly so.
});

// ❌ NestJS: bumping agentVersion on existing conversations during a switch
await tx.conversation.updateMany({
  where: { channelConnectionId },
  data: { agentVersion: { increment: 1 } },  // This was the original bug. Never.
});

// ❌ NestJS: writing isActive=true without revoking the current row in the same tx
await prisma.agentAssignment.create({
  data: { channelConnectionId, agentCodeName: 'sofia', isActive: true },
});  // Will violate the partial unique index. Use the switchAgent transaction.
```

```python
# ❌ Python: silent fallback to a default agent
def get_graph(code_name: str):
    return CODE_NAME_REGISTRY.get(code_name) or CODE_NAME_REGISTRY['helena']  # No.

# ❌ Python: skipping the lock on first compile
if code_name not in _compiled_graphs:
    _compiled_graphs[code_name] = builder(checkpointer)  # Race under concurrent first-hit.

# ❌ Python: hand-building thread IDs
thread_id = f"{tenant_id}_{conversation_id}_v{version}"  # Use scoped_thread_id() always.

# ❌ Python: routing inside a node based on agent_code_name
if state["agent_code_name"] == "sofia":
    return school_specific_thing()  # Wrong graph — the registry already chose for you.
```

```typescript
// ❌ Frontend: skipping the displayName confirmation in the switch dialog
this.store.switchAgent(connId, { codeName: 'sofia' }).subscribe();  // No typed confirmation.
```

</anti-patterns>

---

## Verification commands

Run these before merging any change to the files listed in the skill description. Each command should match the expected result in the comment.

<verification>

### NestJS backend

```bash
# 1. The Prisma snapshot guard is installed and untouched
grep -n "installConversationSnapshotGuard" verveux-backend/src/prisma/prisma.service.ts
# expect: function defined + called from onModuleInit, protected fields list includes
#         agentCodeName, agentAssignmentId, agentVersion

# 2. No code path mutates the snapshot fields outside creation
grep -rn "agentCodeName\|agentAssignmentId\|agentVersion" verveux-backend/src \
  | grep -E "update|upsert" | grep -v "// allowed:"
# expect: zero hits — every update path must have a `// allowed:` waiver comment

# 3. switchAgent runs inside a transaction
grep -A 3 "switchAgent" verveux-backend/src/agent-assignments/agent-assignments.service.ts \
  | grep "\$transaction"
# expect: matches

# 4. ChannelConnectionsService.update rejects agentType
grep -A 5 "async update" verveux-backend/src/channels/channel-connections.service.ts \
  | grep "agentType.*BadRequest\|deprecated"
# expect: explicit 400 with deprecation message

# 5. Migrations: partial unique index exists
grep -rn "agent_assignments_one_active_per_connection" verveux-backend/prisma/migrations
# expect: one migration creates it; no migration drops it

# 6. Type-check + tests
cd verveux-backend && npx tsc -p tsconfig.build.json --noEmit && npm test
```

### Python agent

```bash
# 7. Registry is the only path to a graph
grep -rn "CODE_NAME_REGISTRY\[" verveux-agent/src \
  | grep -v "registry.py"
# expect: zero hits — all access goes through get_or_compile_graph()

# 8. No silent fallback to a default code name
grep -rn "CODE_NAME_REGISTRY.get\|registry.get" verveux-agent/src
# expect: only get_or_compile_graph() and it raises UnknownCodeNameError on miss

# 9. Thread IDs only built via the helper
grep -rn "scoped_thread_id" verveux-agent/src \
  | grep -v "auth/service_auth.py"
# expect: only callers; the definition is in service_auth.py

grep -rn 'f".*:.*:.*:.*:v' verveux-agent/src
# expect: zero hits — no hand-built five-segment thread IDs

# 10. Startup validation is fatal on DB-extra code names
grep -B 2 -A 5 "missing_in_registry" verveux-agent/src/main.py
# expect: raise SystemExit(...) on the missing_in_registry branch

# 11. All metrics carry the agent_code_name label
grep -E "Counter|Histogram" verveux-agent/src/observability.py \
  | grep -v "agent_code_name\|legacy_agent_type_fallback_total"
# expect: zero hits (legacy_agent_type_fallback_total is unlabeled by design — it's a canary)

# 12. The registry test suite
cd verveux-agent && uv run pytest tests/test_registry.py -v
# expect: 8 tests pass — registry seeding, unknown raises, lock under concurrency,
#         warm-up subset behavior
```

### Frontend

```bash
# 13. Switch dialog still requires typed confirmation
grep -A 3 "canSubmit" verveux-frontend/src/app/features/channels/dialogs/switch-agent-dialog.component.ts \
  | grep "confirmText\|displayName"
# expect: canSubmit gated by confirmText().trim().toLowerCase() === selectedDisplayName()...

# 14. Selectable list filters by isActive + tenant availability
grep -A 5 "selectableCodeNames" verveux-frontend/src/app/features/channels/dialogs/switch-agent-dialog.component.ts \
  | grep "isActive\|availableAgentTypes"
# expect: both filters present
```

</verification>

---

## Adding a new agent (safe path)

Use this recipe when introducing a new code name (e.g. `lucia` for `SCHOOL`). It is the only path that respects all eight invariants without manual coordination.

<adding-new-agent>

1. **Add the graph file.** `verveux-agent/src/graphs/lucia_graph.py` exports `build_lucia_graph(checkpointer)`. Do not import or modify other agents' graph files.

2. **Register the code name in Python.** Add one line to `CODE_NAME_REGISTRY` in `verveux-agent/src/graphs/registry.py`:
   ```python
   CODE_NAME_REGISTRY: dict[str, GraphBuilder] = {
       "helena": build_sales_graph,
       "sofia": build_school_graph,
       "lucia": build_lucia_graph,  # ← new
       ...
   }
   ```
   Do not touch existing entries.

3. **Add a database migration.** `verveux-backend/prisma/migrations/<ts>_add_agent_code_name_lucia/migration.sql`:
   ```sql
   INSERT INTO "agent_code_names" ("codeName", "agentType", "displayName", "isActive", "createdAt")
   VALUES ('lucia', 'SCHOOL', 'Lucia', true, NOW())
   ON CONFLICT ("codeName") DO NOTHING;
   ```
   Idempotent. No UPDATE. No DELETE on other rows.

4. **Deploy order matters.** Backend migration first → Python agent rebuild → backend rebuild. If you flip the order, the Python service may start, see `lucia` is missing from `agent_code_names`, treat it as a registry-extra, and warn (harmless). The reverse — `lucia` in the DB but missing from the registry — is fatal by design.

5. **Tenants opt in.** A tenant only sees `lucia` in the switch dialog if `SCHOOL` is in their `TenantSettings.availableAgents`. The switch transaction also re-checks this — there is no path to assign `lucia` to a connection whose tenant has not enabled `SCHOOL`.

6. **Verify before merging:**
   ```bash
   # Python: lucia registered + isolated
   grep -c "lucia" verveux-agent/src/graphs/registry.py
   # expect: 1 (only the new entry)

   cd verveux-agent && uv run pytest tests/test_registry.py -v
   # expect: existing 8 tests still pass; if you add a test for lucia, it runs in isolation

   # Backend: migration is idempotent
   psql "$DATABASE_URL" -c "SELECT \"codeName\" FROM agent_code_names WHERE \"codeName\" = 'lucia';"
   # expect: one row, isActive = true

   # Frontend: dropdown picks it up automatically — no code change required if you
   # follow steps 1-5. If you find yourself editing channel-connection.model.ts or
   # the switch dialog to add lucia, you are off the safe path.
   ```

</adding-new-agent>

---

## Deprecating an existing agent (phase-2 path)

Removing a code name is harder than adding one because of the historical audit trail.

<deprecating-agent>

- **Never delete the row** from `agent_code_names`. The FK from `AgentAssignment` is `Restrict` and will block you — that's intentional.
- **Flip `isActive = false`.** New switches to this code name become impossible. Existing assignments keep working until the admin switches them off manually.
- **Optionally remove the builder from `CODE_NAME_REGISTRY`** only after every active assignment has been switched away. Cross-check with:
  ```sql
  SELECT DISTINCT "agentCodeName"
  FROM "channel_connections"
  WHERE "isActive" = true AND "deletedAt" IS NULL AND "agentCodeName" IS NOT NULL;
  ```
  Every row must already be off the deprecated code name. Otherwise removing the builder will make the next service startup fail (invariant #5).
- **Tests for the deprecated code name stay.** Historical correctness is part of what we promise.

</deprecating-agent>

---

## When to consult this skill

Auto-invoke whenever you are editing any of:

- `verveux-agent/src/graphs/registry.py`
- `verveux-agent/src/main.py` (lifespan, `/chat/stream`, `/chat/resume`)
- `verveux-agent/src/auth/service_auth.py` (`scoped_thread_id`)
- `verveux-agent/src/graphs/state.py`
- `verveux-agent/src/observability.py`
- `verveux-backend/prisma/schema.prisma` (the four affected models)
- `verveux-backend/prisma/migrations/**` (any new migration touching the four models)
- `verveux-backend/src/agent-assignments/**`
- `verveux-backend/src/channels/channel-connections.service.ts`
- `verveux-backend/src/conversations/conversations.service.ts`
- `verveux-backend/src/prisma/prisma.service.ts`
- `verveux-backend/src/agent/agent.service.ts` (thread-ID construction, code-name resolution)
- `verveux-frontend/src/app/features/channels/dialogs/switch-agent-dialog.component.ts`
- `verveux-frontend/src/app/features/channels/data-access/channels-api.service.ts`
- `verveux-frontend/src/app/features/channels/data-access/channels.store.ts`

If the change spans two or more of these files, run the **full** verification block before merging — not just the section for one service.
