package com.fractalov.backend.service.persistence;

public class NotFoundException extends RuntimeException {
    public NotFoundException(String resource, Object id) {
        super(resource + " not found: " + id);
    }
}
