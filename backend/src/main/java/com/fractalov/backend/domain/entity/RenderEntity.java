package com.fractalov.backend.domain.entity;

import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.domain.Persistable;
import org.springframework.data.relational.core.mapping.Table;

import java.time.Instant;
import java.util.UUID;

@Table("renders")
public record RenderEntity(
        @Id UUID id,
        UUID recipeId,
        String imagePath,
        int widthPx,
        int heightPx,
        String paletteName,
        String colorMode,
        short samplesPerAxis,
        long renderMs,
        long colorizeMs,
        long encodeMs,
        long totalMs,
        long fileSizeBytes,
        @CreatedDate Instant createdAt
) implements Persistable<UUID> {

    @Override
    public UUID getId() {
        return id;
    }

    @Override
    public boolean isNew() {
        return createdAt == null;
    }
}
