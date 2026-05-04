package com.fractalov.backend.dto;

import jakarta.validation.constraints.NotBlank;

public record ColorSettings(
        @NotBlank(message = "paletteName must not be blank")
        String paletteName,

        ColorMode mode
) {

    public ColorSettings(String paletteName) {
        this(paletteName, null);
    }

    public ColorMode effectiveMode() {
        return mode == null ? ColorMode.LINEAR : mode;
    }
}
