package com.fractalov.backend.api.dto;

import com.fractalov.backend.dto.FractalRecipe;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;

public final class MlRequests {
    private MlRequests() {}

    public record VariationsRequest(
            @NotNull @Valid FractalRecipe recipe,
            @Min(1) @Max(16) int count,
            int seed
    ) {}
}
