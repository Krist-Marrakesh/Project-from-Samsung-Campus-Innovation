package com.fractalov.backend.dto;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Locale;

public enum ColorMode {
    /** Normalise each escape value by {@code maxIter} and look up the palette. */
    LINEAR,
    /**
     * Rank all escape values by their CDF percentile, then look up the palette.
     * Removes banding in smooth gradients where most pixels escape in a narrow
     * iteration range.
     */
    HISTOGRAM,
    /**
     * Use the per-pixel distance-estimate field rather than escape time. The
     * distance is normalised against the viewport's pixel size and run through
     * a {@code tanh} ramp so values close to the boundary darken smoothly while
     * deep-exterior regions saturate to the palette tail.
     *
     * <p>Falls back to {@link #LINEAR} when the renderer for the recipe's
     * fractal type does not produce a DE field (e.g. Burning Ship: the map is
     * non-holomorphic and the standard estimator does not apply).
     */
    DISTANCE_ESTIMATE;

    @JsonValue
    public String asJson() {
        return name().toLowerCase(Locale.ROOT);
    }

    @JsonCreator
    public static ColorMode fromJson(String value) {
        if (value == null) return LINEAR;
        return valueOf(value.toUpperCase(Locale.ROOT));
    }
}
