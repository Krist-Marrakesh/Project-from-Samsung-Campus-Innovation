package com.fractalov.backend.api;

import com.fractalov.backend.api.dto.RenderJobResponses;
import com.fractalov.backend.config.JobsProperties;
import com.fractalov.backend.domain.entity.JobStatus;
import com.fractalov.backend.domain.entity.RenderJobEntity;
import com.fractalov.backend.service.jobs.JobEventBus;
import com.fractalov.backend.service.jobs.JobLifecycleEvent;
import com.fractalov.backend.service.jobs.RenderJobService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

@RestController
public class RenderJobController {

    private static final Logger log = LoggerFactory.getLogger(RenderJobController.class);

    private final RenderJobService service;
    private final JobEventBus eventBus;
    private final long sseHeartbeatMs;

    public RenderJobController(RenderJobService service, JobEventBus eventBus, JobsProperties props) {
        this.service = service;
        this.eventBus = eventBus;
        this.sseHeartbeatMs = props.sseHeartbeatSeconds() * 1000L;
    }

    @PostMapping("/recipes/{recipeId}/render-jobs")
    public ResponseEntity<RenderJobResponses.View> submit(@PathVariable UUID recipeId) {
        RenderJobEntity job = service.submit(recipeId);
        return ResponseEntity.status(HttpStatus.ACCEPTED).body(RenderJobResponses.View.of(job));
    }

    @GetMapping("/render-jobs/{id}")
    public RenderJobResponses.View get(@PathVariable UUID id) {
        return RenderJobResponses.View.of(service.get(id));
    }

    @GetMapping("/recipes/{recipeId}/render-jobs")
    public List<RenderJobResponses.View> listByRecipe(@PathVariable UUID recipeId) {
        return service.listByRecipe(recipeId).stream()
                .map(RenderJobResponses.View::of)
                .toList();
    }

    @PostMapping("/render-jobs/{id}/cancel")
    public RenderJobResponses.View cancel(@PathVariable UUID id) {
        service.cancel(id);
        return RenderJobResponses.View.of(service.get(id));
    }

    /**
     * SSE stream of job state. Always opens with a {@code snapshot} event so the
     * client sees current state immediately; if the job is already terminal the
     * emitter completes right after that snapshot. Otherwise it subscribes to the
     * in-process event bus and forwards lifecycle events until terminal.
     *
     * <p>Heartbeats every {@code sseHeartbeatSeconds} keep proxies / load balancers
     * from killing idle connections.
     */
    @GetMapping(path = "/render-jobs/{id}/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@PathVariable UUID id) {
        SseEmitter emitter = new SseEmitter(0L);   // 0 = no timeout from Spring side
        AtomicReference<JobEventBus.Subscription> subRef = new AtomicReference<>();

        Runnable cleanup = () -> {
            JobEventBus.Subscription s = subRef.getAndSet(null);
            if (s != null) s.close();
        };
        emitter.onCompletion(cleanup);
        emitter.onTimeout(cleanup);
        emitter.onError(t -> cleanup.run());

        try {
            RenderJobEntity initial = service.get(id);
            sendSnapshot(emitter, initial);
            if (initial.status().isTerminal()) {
                emitter.complete();
                return emitter;
            }
        } catch (Exception ex) {
            emitter.completeWithError(ex);
            return emitter;
        }

        // Subscribe before any further work — events arriving between the snapshot
        // and the subscribe call would be lost otherwise. The bus accepts duplicates
        // (snapshot may overlap the next published event), and the client just sees
        // an idempotent state replay.
        JobEventBus.Subscription sub = eventBus.subscribe(id, ev -> deliver(emitter, ev));
        subRef.set(sub);

        return emitter;
    }

    private void sendSnapshot(SseEmitter emitter, RenderJobEntity entity) throws IOException {
        emitter.send(SseEmitter.event()
                .name("snapshot")
                .data(RenderJobResponses.View.of(entity)));
    }

    private void deliver(SseEmitter emitter, JobLifecycleEvent ev) {
        try {
            emitter.send(SseEmitter.event()
                    .name(ev.status().asJson())
                    .data(ev));
            if (ev.terminal()) {
                emitter.complete();
            }
        } catch (Exception ex) {
            log.debug("SSE delivery failed for job {} ({}); closing", ev.jobId(), ex.toString());
            emitter.completeWithError(ex);
        }
    }
}
