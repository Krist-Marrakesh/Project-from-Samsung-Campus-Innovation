package com.fractalov.backend.dto;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Positive;

public record MultibrotParams(
        @Min(value = 2, message = "exponent must be >= 2")
        @Max(value = 10, message = "exponent must be <= 10")
        int exponent,

        @Min(value = 1, message = "maxIter must be >= 1")
        @Max(value = 10000, message = "maxIter must be <= 10000")
        int maxIter,

        @Positive(message = "escapeRadius must be > 0")
        double escapeRadius,

        boolean smoothing
) implements FractalParams {

    @Override
    @JsonIgnore
    public FractalType fractalType() {
        return FractalType.MULTIBROT;
    }
}
