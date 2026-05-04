package com.fractalov.backend.dto;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;

public record FractalRecipe(
        @NotNull @Valid Viewport viewport,
        @NotNull @Valid RenderSettings renderSettings,
        @NotNull @Valid ColorSettings colorSettings,

        @NotNull
        @Valid
        @JsonTypeInfo(
                use = JsonTypeInfo.Id.NAME,
                include = JsonTypeInfo.As.EXTERNAL_PROPERTY,
                property = "fractalType"
        )
        @JsonSubTypes({
                @JsonSubTypes.Type(value = MandelbrotParams.class, name = "mandelbrot"),
                @JsonSubTypes.Type(value = JuliaParams.class, name = "julia"),
                @JsonSubTypes.Type(value = BurningShipParams.class, name = "burning_ship"),
                @JsonSubTypes.Type(value = MultibrotParams.class, name = "multibrot")
        })
        FractalParams params
) {
    public FractalType fractalType() {
        return params == null ? null : params.fractalType();
    }
}
