package com.fractalov.backend.service.jobs;

import com.fractalov.backend.domain.repo.RenderJobRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;

/**
 * On every boot, sweep any QUEUED/RUNNING rows from the previous process and mark
 * them FAILED. A RUNNING job has no live worker after a kill, and a QUEUED job
 * older than the boot has been waiting suspiciously long anyway — fail fast and
 * let the client retry.
 *
 * <p>This is the simplest workable recovery model. A multi-instance setup would
 * replace this with a heartbeat ({@code last_seen_at}) and only fence rows whose
 * lease expired.
 */
@Component
public class RestartRecoveryRunner implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(RestartRecoveryRunner.class);

    private final RenderJobRepository jobs;

    public RestartRecoveryRunner(RenderJobRepository jobs) {
        this.jobs = jobs;
    }

    @Override
    @Transactional
    public void run(ApplicationArguments args) {
        int n = jobs.markStuckAsFailed(Instant.now(), "server restart");
        if (n > 0) log.info("restart recovery: fenced {} stuck job(s) as failed", n);
    }
}
