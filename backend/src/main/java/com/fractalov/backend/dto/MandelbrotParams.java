package com.fractalov.backend.dto;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Positive;

public record MandelbrotParams(
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
        return FractalType.MANDELBROT;
    }
}
