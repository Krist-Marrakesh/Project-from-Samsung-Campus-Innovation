package com.fractalov.backend.service.ml;

public class MlUnavailableException extends RuntimeException {
    public MlUnavailableException(String message) {
        super(message);
    }

    public MlUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
