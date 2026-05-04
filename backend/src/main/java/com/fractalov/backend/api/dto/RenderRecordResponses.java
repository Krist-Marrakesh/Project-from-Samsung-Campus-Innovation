package com.fractalov.backend.api.dto;

import com.fractalov.backend.domain.entity.RenderEntity;
import com.fractalov.backend.dto.PerformanceBreakdown;

import java.time.Instant;
import java.util.UUID;

public final class RenderRecordResponses {
    private RenderRecordResponses() {}

    public record View(
            UUID id,
            UUID recipeId,
            String imageUrl,
            int widthPx,
            int heightPx,
            String paletteName,
            String colorMode,
            int samplesPerAxis,
            long fileSizeBytes,
            PerformanceBreakdown performance,
            Instant createdAt,
            String imageBase64
    ) {
        public static View of(RenderEntity r, PerformanceBreakdown perf, String imageBase64) {
            return new View(
                    r.id(),
                    r.recipeId(),
                    "/renders/" + r.id() + "/image",
                    r.widthPx(),
                    r.heightPx(),
                    r.paletteName(),
                    r.colorMode(),
                    (int) r.samplesPerAxis(),
                    r.fileSizeBytes(),
                    perf,
                    r.createdAt(),
                    imageBase64
            );
        }
    }
}
