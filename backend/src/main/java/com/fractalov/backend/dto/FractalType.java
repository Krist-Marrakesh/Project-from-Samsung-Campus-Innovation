package com.fractalov.backend.dto;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Locale;

public enum FractalType {
    MANDELBROT,
    JULIA,
    BURNING_SHIP,
    MULTIBROT;

    @JsonValue
    public String asJson() {
        return name().toLowerCase(Locale.ROOT);
    }

    @JsonCreator
    public static FractalType fromJson(String value) {
        if (value == null) {
            throw new IllegalArgumentException("fractalType is required");
        }
        return valueOf(value.toUpperCase(Locale.ROOT));
    }

    /** Matches Jackson subtype names — snake_case, lowercase. */
    public String jsonName() {
        return asJson();
    }
}
