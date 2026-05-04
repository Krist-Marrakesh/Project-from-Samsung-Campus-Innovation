package com.fractalov.backend.api.dto;

import com.fractalov.backend.dto.FractalRecipe;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

public final class RecipeRequests {
    private RecipeRequests() {}

    public record Create(
            @Size(max = 200) String name,
            @NotNull @Valid FractalRecipe recipe
    ) {}
}
