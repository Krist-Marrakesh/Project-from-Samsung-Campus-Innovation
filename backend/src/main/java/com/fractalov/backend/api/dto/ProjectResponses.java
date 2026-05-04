package com.fractalov.backend.api.dto;

import com.fractalov.backend.domain.entity.ProjectEntity;

import java.time.Instant;
import java.util.UUID;

public final class ProjectResponses {
    private ProjectResponses() {}

    public record View(
            UUID id,
            String name,
            String description,
            String ownerId,
            long recipeCount,
            Instant createdAt,
            Instant updatedAt
    ) {
        public static View of(ProjectEntity p, long recipeCount) {
            return new View(p.id(), p.name(), p.description(), p.ownerId(),
                    recipeCount, p.createdAt(), p.updatedAt());
        }
    }
}
