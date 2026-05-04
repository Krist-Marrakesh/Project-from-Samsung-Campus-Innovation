package com.fractalov.backend.dto;

public record RenderResponse(
        String requestId,
        String status,
        String imageBase64,
        String format,
        int widthPx,
        int heightPx,
        PerformanceBreakdown performance,
        FractalRecipe recipeEcho
) {}
