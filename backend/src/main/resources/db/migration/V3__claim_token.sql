-- Stage-9 housekeeping: replace started_at-based claim handoff with an explicit
-- token column.
--
-- Why: RenderJobPoller.claimOne(now) used to UPDATE the claimed row with
-- started_at = :now and then re-fetch via findJustClaimed(now), filtering by
-- started_at = :now. That works on a single instance but breaks the moment
-- two pollers (multi-instance, or a second @Scheduled bean) call claimOne
-- inside the same Postgres millisecond — both UPDATEs land with identical
-- started_at, the re-fetch is non-deterministic.
--
-- Fix: every claim attempt generates a fresh UUID at the call site; UPDATE
-- writes (started_at, claim_token); the re-fetch filters by claim_token. UUID
-- collisions are 2^-122 per pair, which is the canonical "won't happen" bound.

ALTER TABLE render_jobs
    ADD COLUMN IF NOT EXISTS claim_token UUID;

-- The poller scans for the just-claimed row by claim_token. Only one row
-- ever carries any given token, so a btree index on the column gives O(log n)
-- single-row lookup.
CREATE INDEX IF NOT EXISTS idx_render_jobs_claim_token
    ON render_jobs (claim_token)
    WHERE claim_token IS NOT NULL;
