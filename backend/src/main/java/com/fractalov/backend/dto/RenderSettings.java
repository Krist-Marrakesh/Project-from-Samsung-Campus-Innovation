package com.fractalov.backend.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;

public record RenderSettings(
        @Min(value = 1, message = "widthPx must be >= 1")
        @Max(value = 8192, message = "widthPx must be <= 8192")
        int widthPx,

        @Min(value = 1, message = "heightPx must be >= 1")
        @Max(value = 8192, message = "heightPx must be <= 8192")
        int heightPx,

        @Min(value = 1, message = "samplesPerAxis must be >= 1")
        @Max(value = 3, message = "samplesPerAxis must be <= 3")
        Integer samplesPerAxis
) {

    public RenderSettings(int widthPx, int heightPx) {
        this(widthPx, heightPx, null);
    }

    public int effectiveSamplesPerAxis() {
        return samplesPerAxis == null ? 1 : samplesPerAxis;
    }
}
