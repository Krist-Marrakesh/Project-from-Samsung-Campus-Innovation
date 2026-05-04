package com.fractalov.backend.domain.entity;

import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.domain.Persistable;
import org.springframework.data.relational.core.mapping.Table;

import java.time.Instant;
import java.util.UUID;

@Table("projects")
public record ProjectEntity(
        @Id UUID id,
        String name,
        String description,
        String ownerId,
        @CreatedDate Instant createdAt,
        @LastModifiedDate Instant updatedAt
) implements Persistable<UUID> {

    @Override
    public UUID getId() {
        return id;
    }

    /**
     * INSERT vs UPDATE discriminator. {@code createdAt} is populated by the auditing
     * callback only when persisting a fresh record, so a null value here always
     * means "new aggregate". Using id-nullness as the check would force callers to
     * pass null and patch later, which doesn't fit immutable records.
     */
    @Override
    public boolean isNew() {
        return createdAt == null;
    }

    public static ProjectEntity newProject(String name, String description, String ownerId) {
        return new ProjectEntity(UUID.randomUUID(), name, description,
                ownerId == null ? "anonymous" : ownerId, null, null);
    }

    public ProjectEntity withUpdated(String name, String description) {
        return new ProjectEntity(id, name, description, ownerId, createdAt, updatedAt);
    }
}
