package com.fractalov.backend.api.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public final class ProjectRequests {
    private ProjectRequests() {}

    public record Create(
            @NotBlank @Size(max = 200) String name,
            @Size(max = 4000) String description,
            @Size(max = 100) String ownerId
    ) {}

    public record Update(
            @NotBlank @Size(max = 200) String name,
            @Size(max = 4000) String description
    ) {}
}
