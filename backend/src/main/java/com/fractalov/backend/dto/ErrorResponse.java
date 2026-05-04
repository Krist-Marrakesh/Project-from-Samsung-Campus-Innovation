package com.fractalov.backend.dto;

import java.util.List;

public record ErrorResponse(
        String requestId,
        String error,
        String message,
        List<ValidationResponse.FieldViolation> details
) {}
