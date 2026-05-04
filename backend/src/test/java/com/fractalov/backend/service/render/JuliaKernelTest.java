package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.JuliaParams;
import com.fractalov.backend.service.render.kernel.JuliaKernel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JuliaKernelTest {

    @Test
    void deOnEscapingPointIsFiniteAndPositive() {
        // Classic Julia set; sample a point that should escape.
        JuliaKernel k = new JuliaKernel(
                new JuliaParams(-0.7, 0.27015, 500, 2.0, true), true);
        FieldsOut out = new FieldsOut();
        out.reset();
        k.sample(1.5, 1.5, out);
        assertTrue(out.escapeTime >= 0.0, "expected escape");
        assertFalse(Double.isNaN(out.distanceEstimate));
        assertTrue(out.distanceEstimate > 0.0);
    }

    @Test
    void deOnInSetPointIsNaN() {
        JuliaKernel k = new JuliaKernel(
                new JuliaParams(0.0, 0.0, 500, 2.0, false), true);
        FieldsOut out = new FieldsOut();
        out.reset();
        k.sample(0.0, 0.0, out);
        assertEquals(-1.0, out.escapeTime, 1e-12);
        assertTrue(Double.isNaN(out.distanceEstimate));
    }
}
