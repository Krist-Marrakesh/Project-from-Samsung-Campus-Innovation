package com.fractalov.backend.dto;

import java.util.List;

public record ValidationResponse(
        boolean valid,
        List<FieldViolation> errors
) {
    public record FieldViolation(String field, String message) {}
}
