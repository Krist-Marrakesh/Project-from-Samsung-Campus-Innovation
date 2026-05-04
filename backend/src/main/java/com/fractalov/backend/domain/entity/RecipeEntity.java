package com.fractalov.backend.domain.entity;

import com.fractalov.backend.dto.FractalRecipe;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.domain.Persistable;
import org.springframework.data.relational.core.mapping.Table;

import java.time.Instant;
import java.util.UUID;

@Table("recipes")
public record RecipeEntity(
        @Id UUID id,
        UUID projectId,
        String name,
        String fractalType,
        FractalRecipe recipeJson,
        int version,
        @CreatedDate Instant createdAt
) implements Persistable<UUID> {

    @Override
    public UUID getId() {
        return id;
    }

    @Override
    public boolean isNew() {
        return createdAt == null;
    }

    public static RecipeEntity newRecipe(UUID projectId, String name, FractalRecipe recipe) {
        return new RecipeEntity(
                UUID.randomUUID(),
                projectId,
                name,
                recipe.fractalType().asJson(),
                recipe,
                1,
                null
        );
    }
}
