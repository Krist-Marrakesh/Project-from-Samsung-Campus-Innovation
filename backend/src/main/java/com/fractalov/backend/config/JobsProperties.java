package com.fractalov.backend.config;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties(prefix = "app.jobs")
public record JobsProperties(
        @Min(1) @Max(32) int workerPoolSize,
        @Min(50) @Max(60_000) long pollIntervalMs,
        @Min(1) @Max(60) int sseHeartbeatSeconds
) {
    public JobsProperties {
        if (workerPoolSize == 0) workerPoolSize = 1;
        if (pollIntervalMs == 0) pollIntervalMs = 500;
        if (sseHeartbeatSeconds == 0) sseHeartbeatSeconds = 15;
    }
}
