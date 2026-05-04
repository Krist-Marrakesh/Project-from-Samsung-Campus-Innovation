package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.MandelbrotParams;
import com.fractalov.backend.service.render.kernel.MandelbrotKernel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Direct kernel-level tests — assert properties that are awkward to assert
 * from the renderer level (per-call output buffer, internal short-circuits,
 * derivative track behaviour).
 */
class MandelbrotKernelTest {

    private final MandelbrotParams stdParams = new MandelbrotParams(500, 2.0, true);

    @Test
    void cardioidShortCircuitClassifiesOriginAsInSet() {
        // c = 0 sits inside the main cardioid; the kernel must report -1.0 without
        // running the iteration loop. We can't directly observe iteration count,
        // but we can verify the result and trust the code path.
        MandelbrotKernel k = new MandelbrotKernel(stdParams, false);
        FieldsOut out = new FieldsOut();
        out.reset();
        k.sample(0.0, 0.0, out);
        assertEquals(-1.0, out.escapeTime, 1e-12);
    }

    @Test
    void period2BulbCenterIsInSet() {
        // (-1, 0) is the centre of the period-2 bulb, also caught analytically.
        MandelbrotKernel k = new MandelbrotKernel(stdParams, false);
        FieldsOut out = new FieldsOut();
        out.reset();
        k.sample(-1.0, 0.0, out);
        assertEquals(-1.0, out.escapeTime, 1e-12);
    }

    @Test
    void farPointEscapesQuickly() {
        MandelbrotKernel k = new MandelbrotKernel(stdParams, false);
        FieldsOut out = new FieldsOut();
        out.reset();
        k.sample(4.0, 4.0, out);
        assertTrue(out.escapeTime >= 0.0 && out.escapeTime < 5.0,
                "far point should escape near iteration 0, got " + out.escapeTime);
    }

    @Test
    void deOptOutLeavesDistanceAsNaN() {
        MandelbrotKernel k = new MandelbrotKernel(stdParams, false);
        FieldsOut out = new FieldsOut();
        out.reset();
        k.sample(-0.75, 0.1, out);   // border-near point that escapes
        assertTrue(out.escapeTime >= 0.0, "expected this border point to escape");
        assertTrue(Double.isNaN(out.distanceEstimate),
                "DE must be NaN when computeDe=false, got " + out.distanceEstimate);
    }

    @Test
    void deOnEscapingPointIsFiniteAndPositive() {
        MandelbrotKernel k = new MandelbrotKernel(stdParams, true);
        FieldsOut out = new FieldsOut();
        out.reset();
        k.sample(-0.75, 0.25, out);
        assertTrue(out.escapeTime >= 0.0, "expected escape, got " + out.escapeTime);
        assertFalse(Double.isNaN(out.distanceEstimate));
        assertTrue(out.distanceEstimate > 0.0,
                "DE must be positive on escape, got " + out.distanceEstimate);
        assertTrue(Double.isFinite(out.distanceEstimate),
                "DE must be finite, got " + out.distanceEstimate);
    }

    @Test
    void deOnInSetPointIsNaN() {
        MandelbrotKernel k = new MandelbrotKernel(stdParams, true);
        FieldsOut out = new FieldsOut();
        out.reset();
        k.sample(0.0, 0.0, out);
        assertEquals(-1.0, out.escapeTime, 1e-12);
        assertTrue(Double.isNaN(out.distanceEstimate),
                "in-set DE must be NaN, got " + out.distanceEstimate);
    }

    @Test
    void deDecreasesAsBoundaryApproaches() {
        // Standard property: DE shrinks monotonically as c approaches the boundary.
        // Sample two points on the same horizontal line moving outward from the
        // body toward the exterior; the closer-to-boundary one should have smaller DE.
        MandelbrotKernel k = new MandelbrotKernel(stdParams, true);
        FieldsOut a = new FieldsOut();
        FieldsOut b = new FieldsOut();
        a.reset();
        b.reset();
        k.sample(0.45, 0.0, a);    // close to right edge of cardioid
        k.sample(2.0, 0.0, b);     // far exterior
        assertTrue(a.escapeTime >= 0.0);
        assertTrue(b.escapeTime >= 0.0);
        assertTrue(a.distanceEstimate < b.distanceEstimate,
                "DE near boundary (" + a.distanceEstimate + ") should be smaller than DE far away ("
                        + b.distanceEstimate + ")");
    }
}
