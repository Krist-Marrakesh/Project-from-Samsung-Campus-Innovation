package com.fractalov.backend.api.dto;

import com.fractalov.backend.domain.entity.JobStatus;
import com.fractalov.backend.domain.entity.RenderJobEntity;

import java.time.Instant;
import java.util.UUID;

public final class RenderJobResponses {
    private RenderJobResponses() {}

    public record View(
            UUID id,
            UUID recipeId,
            JobStatus status,
            UUID renderId,
            String imageUrl,
            String errorMessage,
            int attemptCount,
            Instant queuedAt,
            Instant startedAt,
            Instant finishedAt
    ) {
        public static View of(RenderJobEntity j) {
            String imageUrl = j.renderId() == null ? null : "/renders/" + j.renderId() + "/image";
            return new View(
                    j.id(),
                    j.recipeId(),
                    j.status(),
                    j.renderId(),
                    imageUrl,
                    j.errorMessage(),
                    (int) j.attemptCount(),
                    j.queuedAt(),
                    j.startedAt(),
                    j.finishedAt()
            );
        }
    }
}
