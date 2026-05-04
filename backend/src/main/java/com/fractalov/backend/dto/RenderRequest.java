package com.fractalov.backend.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;

public record RenderRequest(
        @NotNull @Valid FractalRecipe recipe
) {}
