package com.fractalov.backend.api;

import com.fractalov.backend.dto.HealthResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HealthController {

    private final String version;

    public HealthController(@Value("${spring.application.name:fractalov-backend}") String appName) {
        this.version = "0.0.1-SNAPSHOT";
    }

    @GetMapping("/health")
    public HealthResponse health() {
        return new HealthResponse("ok", version);
    }
}
