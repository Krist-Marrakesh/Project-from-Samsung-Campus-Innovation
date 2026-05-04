package com.fractalov.backend.dto.validation;

import com.fractalov.backend.dto.Viewport;
import jakarta.validation.ConstraintValidator;
import jakarta.validation.ConstraintValidatorContext;

public class ViewportValidator implements ConstraintValidator<ValidViewport, Viewport> {

    private static final double MIN_SPAN = 1e-15;

    @Override
    public boolean isValid(Viewport v, ConstraintValidatorContext ctx) {
        if (v == null) {
            return true;
        }
        boolean finite = Double.isFinite(v.xMin()) && Double.isFinite(v.xMax())
                && Double.isFinite(v.yMin()) && Double.isFinite(v.yMax());
        if (!finite) {
            return false;
        }
        return (v.xMax() - v.xMin()) > MIN_SPAN && (v.yMax() - v.yMin()) > MIN_SPAN;
    }
}
