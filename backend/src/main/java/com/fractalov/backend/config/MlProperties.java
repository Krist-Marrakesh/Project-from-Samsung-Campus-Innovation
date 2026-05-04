package com.fractalov.backend.config;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

/**
 * Connection settings for the Stage 7 Python inference service.
 * <p>The {@code enabled} flag lets deployments run without ML — every
 * {@code /ml/*} endpoint then returns 503 with a descriptive error rather
 * than failing on bean wiring or producing confusing connection-refused
 * stack traces.
 */
@Validated
@ConfigurationProperties(prefix = "app.ml")
public record MlProperties(
        boolean enabled,
        @NotBlank String url,
        @Min(100) long timeoutMs
) {
    public MlProperties {
        if (timeoutMs == 0) timeoutMs = 5_000;
    }
}
