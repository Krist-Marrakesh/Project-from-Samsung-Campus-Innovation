package com.fractalov.backend.service.jobs;

import com.fractalov.backend.config.JobsProperties;
import com.fractalov.backend.domain.entity.RenderJobEntity;
import com.fractalov.backend.domain.repo.RenderJobRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Periodically attempts to claim queued jobs and dispatch them onto the worker pool.
 * Claim semantics: {@link RenderJobRepository#claimOne(Instant, UUID)} runs
 * {@code SELECT … FOR UPDATE SKIP LOCKED}, so multiple instances racing the same
 * queue never grab the same row.
 *
 * <p>Backpressure is by free-slot accounting on the executor: we stop pulling once
 * the pool has no idle threads, instead of buffering jobs in an in-memory queue.
 * This means a queued row stays queued until a worker is free — which is exactly
 * the property we want for fair multi-instance behaviour.
 */
@Component
public class RenderJobPoller {

    private static final Logger log = LoggerFactory.getLogger(RenderJobPoller.class);

    private final RenderJobRepository jobs;
    private final RenderJobWorker worker;
    private final ThreadPoolTaskExecutor executor;
    private final int poolSize;
    private final AtomicInteger inFlight = new AtomicInteger(0);

    public RenderJobPoller(RenderJobRepository jobs,
                           RenderJobWorker worker,
                           @Qualifier("renderJobExecutor") ThreadPoolTaskExecutor executor,
                           JobsProperties props) {
        this.jobs = jobs;
        this.worker = worker;
        this.executor = executor;
        this.poolSize = props.workerPoolSize();
    }

    @Scheduled(fixedDelayString = "${app.jobs.poll-interval-ms:500}")
    public void poll() {
        // Pull as many jobs as we have free worker slots in this tick.
        while (inFlight.get() < poolSize) {
            RenderJobEntity claimed = claimAndFetch();
            if (claimed == null) return;
            inFlight.incrementAndGet();
            executor.execute(() -> {
                try {
                    worker.run(claimed);
                } finally {
                    inFlight.decrementAndGet();
                }
            });
        }
    }

    @Transactional
    protected RenderJobEntity claimAndFetch() {
        // Fresh token per attempt — the UPDATE writes it, the SELECT reads it.
        // UUID collisions are 2^-122, well below "won't happen" threshold.
        UUID token = UUID.randomUUID();
        Instant now = Instant.now();
        int updated = jobs.claimOne(now, token);
        if (updated == 0) return null;
        return jobs.findJustClaimed(token).orElseGet(() -> {
            log.warn("claimOne touched a row but findJustClaimed(token={}) returned empty", token);
            return null;
        });
    }
}
