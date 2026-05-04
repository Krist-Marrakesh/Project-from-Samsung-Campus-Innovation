package com.fractalov.backend.api.dto;

import com.fractalov.backend.domain.entity.RecipeEntity;
import com.fractalov.backend.dto.FractalRecipe;

import java.time.Instant;
import java.util.UUID;

public final class RecipeResponses {
    private RecipeResponses() {}

    public record View(
            UUID id,
            UUID projectId,
            String name,
            String fractalType,
            int version,
            FractalRecipe recipe,
            Instant createdAt
    ) {
        public static View of(RecipeEntity r) {
            return new View(r.id(), r.projectId(), r.name(), r.fractalType(),
                    r.version(), r.recipeJson(), r.createdAt());
        }
    }
}
