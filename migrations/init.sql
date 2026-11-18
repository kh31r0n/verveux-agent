CREATE EXTENSION IF NOT EXISTS vector;

-- RAG document store
CREATE TABLE IF NOT EXISTS documents (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content    TEXT NOT NULL,
    metadata   JSONB NOT NULL DEFAULT '{}',
    embedding  vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Interrupt audit log
CREATE TABLE IF NOT EXISTS approval_requests (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id   TEXT NOT NULL,
    payload     JSONB NOT NULL,
    status      TEXT DEFAULT 'pending',  -- pending | approved | rejected
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS approval_requests_thread_status_idx
    ON approval_requests (thread_id, status);

-- ---------------------------------------------------------------------------
-- LangGraph AsyncPostgresSaver checkpoint tables
-- Mirrors the DDL run by AsyncPostgresSaver.setup() so that the tables exist
-- before the agent starts and checkpointer.setup() becomes a no-op.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id            TEXT NOT NULL,
    checkpoint_ns        TEXT NOT NULL DEFAULT '',
    checkpoint_id        TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type                 TEXT,
    checkpoint           JSONB NOT NULL,
    metadata             JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel       TEXT NOT NULL,
    version       TEXT NOT NULL,
    type          TEXT NOT NULL,
    blob          BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    idx           INTEGER NOT NULL,
    channel       TEXT NOT NULL,
    type          TEXT,
    blob          BYTEA NOT NULL,
    task_path     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx
    ON checkpoints (thread_id);

CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx
    ON checkpoint_blobs (thread_id);

CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx
    ON checkpoint_writes (thread_id);

-- ── Campus (src/campus/): the board game's agents ─────────────────────────────
-- A session's stream of agent events, readable from any instance. See
-- src/campus/store.py. Deleted after CAMPUS_RETENTION_DAYS by the service.
CREATE TABLE IF NOT EXISTS campus_streams (
    id           TEXT PRIMARY KEY,
    epoch        TEXT NOT NULL,
    info         JSONB NOT NULL DEFAULT '{}',
    closed       BOOLEAN NOT NULL DEFAULT false,
    next_offset  INTEGER NOT NULL DEFAULT 0,
    mate_spoke   JSONB NOT NULL DEFAULT '{}',
    mate_seq     INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_feed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS campus_events (
    stream_id  TEXT NOT NULL REFERENCES campus_streams (id) ON DELETE CASCADE,
    "offset"   INTEGER NOT NULL,
    event      TEXT NOT NULL,
    data       JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stream_id, "offset")
);

CREATE INDEX IF NOT EXISTS campus_streams_last_feed_at_idx
    ON campus_streams (last_feed_at);

-- A question design the host asked for; the material is in Cloud Storage.
CREATE TABLE IF NOT EXISTS campus_designs (
    id         TEXT PRIMARY KEY,
    status     TEXT NOT NULL DEFAULT 'working',
    proposal   JSONB NOT NULL DEFAULT '{}',
    approved   JSONB,
    error      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS campus_designs_created_at_idx
    ON campus_designs (created_at);
