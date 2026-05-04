-- Stage 4: async render jobs.
-- DB-as-queue using FOR UPDATE SKIP LOCKED on (status, queued_at).
-- Status state machine:
--   queued -> running -> succeeded
--   queued -> running -> failed
--   queued -> cancelled
--   running -> failed   (set by RestartRecoveryRunner if process died mid-run)

CREATE TABLE IF NOT EXISTS render_jobs (
    id             UUID PRIMARY KEY,
    recipe_id      UUID                       NOT NULL REFERENCES recipes (id) ON DELETE CASCADE,
    status         VARCHAR(20)                NOT NULL,
    render_id      UUID                       REFERENCES renders (id) ON DELETE SET NULL,
    error_message  TEXT,
    attempt_count  SMALLINT                   NOT NULL DEFAULT 0,
    queued_at      TIMESTAMP WITH TIME ZONE   NOT NULL DEFAULT now(),
    started_at     TIMESTAMP WITH TIME ZONE,
    finished_at    TIMESTAMP WITH TIME ZONE,
    -- Spring Data JDBC auditing populates this on insert; needed by Persistable.isNew().
    created_at     TIMESTAMP WITH TIME ZONE   NOT NULL DEFAULT now(),
    CONSTRAINT chk_render_jobs_status CHECK (status IN
        ('queued', 'running', 'succeeded', 'failed', 'cancelled'))
);

-- Drives the poller's claim query. Partial index keeps it tiny — the only rows
-- the poller ever scans are status='queued', and queued ones drain quickly.
CREATE INDEX IF NOT EXISTS idx_render_jobs_queued
    ON render_jobs (queued_at)
    WHERE status = 'queued';

CREATE INDEX IF NOT EXISTS idx_render_jobs_recipe
    ON render_jobs (recipe_id, queued_at DESC);
