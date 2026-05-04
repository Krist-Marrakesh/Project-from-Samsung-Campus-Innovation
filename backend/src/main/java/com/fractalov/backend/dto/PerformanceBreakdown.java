package com.fractalov.backend.dto;

public record PerformanceBreakdown(
        long renderMs,
        long colorizeMs,
        long encodeMs,
        long totalMs
) {}
