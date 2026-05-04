package com.fractalov.backend.domain.entity;

import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.domain.Persistable;
import org.springframework.data.relational.core.mapping.Column;
import org.springframework.data.relational.core.mapping.Table;

import java.time.Instant;
import java.util.UUID;

/**
 * Async render job. Stored row drives the worker pool: rows in {@code QUEUED} state
 * are picked up by {@link com.fractalov.backend.service.jobs.RenderJobPoller} via
 * {@code SELECT ... FOR UPDATE SKIP LOCKED}, transitioned to {@code RUNNING}, and
 * finally to {@code SUCCEEDED}/{@code FAILED} by the worker.
 *
 * <p>{@code status} is stored as plain VARCHAR (not Postgres enum) so we can ship
 * Flyway migrations that add new states without {@code ALTER TYPE} dance.
 *
 * <p>{@code claimToken} is non-null only on rows the poller has just claimed but
 * not yet handed to a worker. It's the handoff key used by
 * {@link com.fractalov.backend.domain.repo.RenderJobRepository#findJustClaimed}
 * — see V3 migration for the rationale.
 */
@Table("render_jobs")
public record RenderJobEntity(
        @Id UUID id,
        UUID recipeId,
        @Column("status") String statusRaw,
        UUID renderId,
        String errorMessage,
        short attemptCount,
        Instant queuedAt,
        Instant startedAt,
        Instant finishedAt,
        UUID claimToken,
        @CreatedDate Instant createdAt
) implements Persistable<UUID> {

    @Override
    public UUID getId() {
        return id;
    }

    @Override
    public boolean isNew() {
        return createdAt == null;
    }

    public JobStatus status() {
        return JobStatus.valueOf(statusRaw.toUpperCase());
    }

    public static RenderJobEntity newQueued(UUID recipeId) {
        return new RenderJobEntity(
                UUID.randomUUID(),
                recipeId,
                JobStatus.QUEUED.asJson(),
                null,
                null,
                (short) 0,
                Instant.now(),
                null,
                null,
                null,
                null
        );
    }

    public Builder toBuilder() {
        return new Builder(this);
    }

    /**
     * Mutability without giving up records. Each setter returns the builder; build()
     * snapshots into a new immutable entity. Used by the worker to express
     * "preserve everything except status + finished_at".
     */
    public static final class Builder {
        private UUID id;
        private UUID recipeId;
        private String statusRaw;
        private UUID renderId;
        private String errorMessage;
        private short attemptCount;
        private Instant queuedAt;
        private Instant startedAt;
        private Instant finishedAt;
        private UUID claimToken;
        private Instant createdAt;

        Builder(RenderJobEntity src) {
            this.id = src.id;
            this.recipeId = src.recipeId;
            this.statusRaw = src.statusRaw;
            this.renderId = src.renderId;
            this.errorMessage = src.errorMessage;
            this.attemptCount = src.attemptCount;
            this.queuedAt = src.queuedAt;
            this.startedAt = src.startedAt;
            this.finishedAt = src.finishedAt;
            this.claimToken = src.claimToken;
            this.createdAt = src.createdAt;
        }

        public Builder status(JobStatus s) { this.statusRaw = s.asJson(); return this; }
        public Builder renderId(UUID v) { this.renderId = v; return this; }
        public Builder errorMessage(String v) { this.errorMessage = v; return this; }
        public Builder finishedAt(Instant v) { this.finishedAt = v; return this; }
        public Builder startedAt(Instant v) { this.startedAt = v; return this; }
        public Builder attemptCount(int v) { this.attemptCount = (short) v; return this; }
        public Builder claimToken(UUID v) { this.claimToken = v; return this; }

        public RenderJobEntity build() {
            return new RenderJobEntity(
                    id, recipeId, statusRaw, renderId, errorMessage, attemptCount,
                    queuedAt, startedAt, finishedAt, claimToken, createdAt
            );
        }
    }
}
