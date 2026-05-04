package com.fractalov.backend.service.jobs;

import com.fractalov.backend.domain.entity.JobStatus;

import java.time.Instant;
import java.util.UUID;

/**
 * Snapshot of a job's state at a moment in time. Published on the {@link JobEventBus}
 * whenever the worker transitions a job and consumed by SSE subscribers.
 */
public record JobLifecycleEvent(
        UUID jobId,
        UUID recipeId,
        JobStatus status,
        UUID renderId,
        String errorMessage,
        Instant at
) {
    public boolean terminal() {
        return status.isTerminal();
    }
}
