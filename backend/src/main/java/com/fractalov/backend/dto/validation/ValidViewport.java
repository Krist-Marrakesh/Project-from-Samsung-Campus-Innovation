package com.fractalov.backend.dto.validation;

import jakarta.validation.Constraint;
import jakarta.validation.Payload;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target({ElementType.TYPE, ElementType.TYPE_USE})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = ViewportValidator.class)
public @interface ValidViewport {
    String message() default "viewport must have xMax > xMin and yMax > yMin with non-degenerate span";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
