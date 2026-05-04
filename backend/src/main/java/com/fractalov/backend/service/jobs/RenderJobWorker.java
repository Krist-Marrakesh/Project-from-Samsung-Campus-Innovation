package com.fractalov.backend.service.jobs;

import com.fractalov.backend.domain.entity.JobStatus;
import com.fractalov.backend.domain.entity.RenderJobEntity;
import com.fractalov.backend.domain.repo.RenderJobRepository;
import com.fractalov.backend.service.persistence.RenderHistoryService;
import com.fractalov.backend.service.persistence.RenderHistoryService.PersistedRender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.task.AsyncTaskExecutor;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.UUID;

/**
 * Runs a single claimed-and-RUNNING job: invokes the existing
 * {@link RenderHistoryService}, then transitions the row to SUCCEEDED or FAILED
 * and publishes the lifecycle event.
 *
 * <p>The compute itself happens on the {@code renderJobExecutor} pool, deliberately
 * separate from request-serving threads so a long-running render can't starve the
 * web tier.
 */
@Component
public class RenderJobWorker {

    private static final Logger log = LoggerFactory.getLogger(RenderJobWorker.class);

    private final RenderJobRepository jobs;
    private final RenderHistoryService renderHistory;
    private final JobEventBus eventBus;
    private final AsyncTaskExecutor executor;

    public RenderJobWorker(RenderJobRepository jobs,
                           RenderHistoryService renderHistory,
                           JobEventBus eventBus,
                           @Qualifier("renderJobExecutor") AsyncTaskExecutor executor) {
        this.jobs = jobs;
        this.renderHistory = renderHistory;
        this.eventBus = eventBus;
        this.executor = executor;
    }

    public void runOnPool(RenderJobEntity claimed) {
        executor.execute(() -> run(claimed));
    }

    void run(RenderJobEntity claimed) {
        UUID jobId = claimed.id();
        UUID recipeId = claimed.recipeId();

        // The poller has already set status=running + started_at; emit a lifecycle
        // event so SSE subscribers see the transition.
        eventBus.publish(new JobLifecycleEvent(
                jobId, recipeId, JobStatus.RUNNING, null, null, Instant.now()));

        try {
            PersistedRender result = renderHistory.renderAndPersist(recipeId, false);
            UUID renderId = result.entity().id();
            jobs.save(claimed.toBuilder()
                    .status(JobStatus.SUCCEEDED)
                    .renderId(renderId)
                    .finishedAt(Instant.now())
                    .build());
            eventBus.publish(new JobLifecycleEvent(
                    jobId, recipeId, JobStatus.SUCCEEDED, renderId, null, Instant.now()));
            log.info("job succeeded id={} recipe={} render={}", jobId, recipeId, renderId);
        } catch (Exception ex) {
            String msg = ex.getMessage();
            if (msg == null) msg = ex.getClass().getSimpleName();
            String trimmed = msg.length() > 1000 ? msg.substring(0, 1000) : msg;
            try {
                jobs.save(claimed.toBuilder()
                        .status(JobStatus.FAILED)
                        .errorMessage(trimmed)
                        .finishedAt(Instant.now())
                        .build());
            } catch (Exception persistFailure) {
                log.error("failed to persist FAILED status for job {}", jobId, persistFailure);
            }
            eventBus.publish(new JobLifecycleEvent(
                    jobId, recipeId, JobStatus.FAILED, null, trimmed, Instant.now()));
            log.warn("job failed id={} recipe={} error={}", jobId, recipeId, trimmed);
        }
    }
}
