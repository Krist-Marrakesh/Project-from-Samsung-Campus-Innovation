package com.fractalov.backend.service.jobs;

import org.springframework.stereotype.Component;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * In-process pub/sub for job lifecycle events. Lives entirely in memory — that is
 * enough for a single-instance backend (Stage 4 scope). Multi-instance setups would
 * swap this out for Postgres LISTEN/NOTIFY or Redis pub/sub without changing the
 * controller's SSE wiring.
 */
@Component
public class JobEventBus {

    private final ConcurrentHashMap<UUID, List<Consumer<JobLifecycleEvent>>> listeners =
            new ConcurrentHashMap<>();

    public Subscription subscribe(UUID jobId, Consumer<JobLifecycleEvent> handler) {
        listeners.computeIfAbsent(jobId, k -> new CopyOnWriteArrayList<>()).add(handler);
        return () -> {
            List<Consumer<JobLifecycleEvent>> list = listeners.get(jobId);
            if (list != null) {
                list.remove(handler);
                if (list.isEmpty()) {
                    listeners.remove(jobId, list);
                }
            }
        };
    }

    public void publish(JobLifecycleEvent event) {
        List<Consumer<JobLifecycleEvent>> list = listeners.get(event.jobId());
        if (list == null) return;
        for (Consumer<JobLifecycleEvent> handler : list) {
            try {
                handler.accept(event);
            } catch (Exception ignored) {
                // A failing subscriber must not affect siblings; the controller layer
                // catches its own emitter errors and unsubscribes via Subscription.
            }
        }
    }

    @FunctionalInterface
    public interface Subscription extends AutoCloseable {
        @Override
        void close();
    }
}
