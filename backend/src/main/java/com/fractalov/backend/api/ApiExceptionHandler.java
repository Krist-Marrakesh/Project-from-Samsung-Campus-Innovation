package com.fractalov.backend.api;

import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.exc.InvalidTypeIdException;
import com.fractalov.backend.dto.ErrorResponse;
import com.fractalov.backend.dto.ValidationResponse.FieldViolation;
import com.fractalov.backend.service.ml.MlUnavailableException;
import com.fractalov.backend.service.persistence.NotFoundException;
import com.fractalov.backend.service.render.RenderException;
import jakarta.validation.ConstraintViolationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.FieldError;
import org.springframework.validation.ObjectError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.List;
import java.util.UUID;

@RestControllerAdvice
public class ApiExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(ApiExceptionHandler.class);

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidation(MethodArgumentNotValidException ex) {
        List<FieldViolation> details = ex.getBindingResult().getAllErrors().stream()
                .map(ApiExceptionHandler::toViolation)
                .toList();
        return badRequest("validation_failed", "Request validation failed", details);
    }

    @ExceptionHandler(ConstraintViolationException.class)
    public ResponseEntity<ErrorResponse> handleConstraintViolation(ConstraintViolationException ex) {
        List<FieldViolation> details = ex.getConstraintViolations().stream()
                .map(v -> new FieldViolation(v.getPropertyPath().toString(), v.getMessage()))
                .toList();
        return badRequest("validation_failed", "Request validation failed", details);
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<ErrorResponse> handleUnreadable(HttpMessageNotReadableException ex) {
        Throwable cause = ex.getCause();
        if (cause instanceof InvalidTypeIdException bad) {
            String message = "unknown fractalType: " + bad.getTypeId();
            return badRequest("unknown_fractal_type", message,
                    List.of(new FieldViolation("recipe.fractalType", message)));
        }
        String message = rootMessage(ex);
        String path = null;
        if (cause instanceof JsonMappingException jme && jme.getPath() != null && !jme.getPath().isEmpty()) {
            StringBuilder sb = new StringBuilder();
            jme.getPath().forEach(ref -> {
                if (ref.getFieldName() != null) {
                    if (sb.length() > 0) sb.append('.');
                    sb.append(ref.getFieldName());
                }
            });
            if (sb.length() > 0) path = sb.toString();
        }
        List<FieldViolation> details = path == null
                ? List.of()
                : List.of(new FieldViolation(path, message));
        return badRequest("malformed_request", message, details);
    }

    @ExceptionHandler(RenderException.class)
    public ResponseEntity<ErrorResponse> handleRender(RenderException ex) {
        return badRequest("render_error", ex.getMessage(), List.of());
    }

    @ExceptionHandler(MlUnavailableException.class)
    public ResponseEntity<ErrorResponse> handleMlUnavailable(MlUnavailableException ex) {
        // ML is best-effort: failures and disablement both surface as 503 so
        // the client can degrade gracefully (continue rendering manually
        // crafted recipes) instead of treating an ML outage as a backend bug.
        ErrorResponse body = new ErrorResponse(
                UUID.randomUUID().toString(),
                "ml_unavailable",
                ex.getMessage(),
                List.of()
        );
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(body);
    }

    @ExceptionHandler(NotFoundException.class)
    public ResponseEntity<ErrorResponse> handleNotFound(NotFoundException ex) {
        ErrorResponse body = new ErrorResponse(
                UUID.randomUUID().toString(),
                "not_found",
                ex.getMessage(),
                List.of()
        );
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(body);
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleIllegalArgument(IllegalArgumentException ex) {
        return badRequest("bad_argument", ex.getMessage(), List.of());
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleOther(Exception ex) {
        String requestId = UUID.randomUUID().toString();
        log.error("unhandled error id={}", requestId, ex);
        ErrorResponse body = new ErrorResponse(
                requestId,
                "internal_error",
                "Internal server error",
                List.of()
        );
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(body);
    }

    private static ResponseEntity<ErrorResponse> badRequest(String code, String message, List<FieldViolation> details) {
        ErrorResponse body = new ErrorResponse(
                UUID.randomUUID().toString(),
                code,
                message,
                details
        );
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(body);
    }

    private static FieldViolation toViolation(ObjectError e) {
        String field;
        if (e instanceof FieldError fe) {
            field = fe.getField().isEmpty() ? fe.getObjectName() : fe.getField();
        } else {
            field = e.getObjectName();
        }
        return new FieldViolation(field, e.getDefaultMessage());
    }

    private static String rootMessage(Throwable t) {
        Throwable cur = t;
        while (cur.getCause() != null && cur.getCause() != cur) {
            cur = cur.getCause();
        }
        return cur.getMessage() == null ? t.getMessage() : cur.getMessage();
    }
}
