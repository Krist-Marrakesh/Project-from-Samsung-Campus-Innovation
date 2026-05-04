package com.fractalov.backend.dto;

import com.fractalov.backend.dto.validation.ValidViewport;

@ValidViewport
public record Viewport(
        double xMin,
        double xMax,
        double yMin,
        double yMax
) {}
