package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.RenderSettings;
import com.fractalov.backend.dto.Viewport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Coverage of the {@link EscapeTimeEngine} core contract: coordinate mapping,
 * SSAA averaging on the escape-time field, and the parallel DE-field path.
 *
 * <p>Each test uses a degenerate {@link IterationKernel} that emits a known
 * function of {@code (cRe, cIm)}, so we can assert exact values without
 * pulling in a full fractal family. The kernels here are not registered as
 * Spring beans and exist only inside the tests.
 */
class EscapeTimeEngineTest {

    private final EscapeTimeEngine engine = new EscapeTimeEngine();

    @Test
    void singleSampleKernelReceivesCornerCoordinates() {
        // Top-left of the 2x2 grid samples (xMin, yMax); bottom-right samples (xMax, yMin).
        IterationKernel k = (cRe, cIm, out) -> { out.escapeTime = cRe + 10 * cIm; };
        FieldStack r = engine.sweep(new Viewport(0.0, 1.0, 0.0, 1.0),
                new RenderSettings(2, 2), k, 1);
        assertEquals(10.0, r.escapeMap()[0][0], 1e-9);
        assertEquals(11.0, r.escapeMap()[0][1], 1e-9);
        assertEquals(0.0, r.escapeMap()[1][0], 1e-9);
        assertEquals(1.0, r.escapeMap()[1][1], 1e-9);
        assertNull(r.distanceMap(), "kernel did not opt in to DE — distance map must stay null");
    }

    @Test
    void ssaaAveragesEscapeValues() {
        IterationKernel k = (cRe, cIm, out) -> { out.escapeTime = 42.0; };
        FieldStack r = engine.sweep(new Viewport(-1.0, 1.0, -1.0, 1.0),
                new RenderSettings(4, 4, 3), k, 1);
        for (double[] row : r.escapeMap()) {
            for (double v : row) assertEquals(42.0, v, 1e-9);
        }
    }

    @Test
    void ssaaPreservesPureInSetPixels() {
        IterationKernel k = (cRe, cIm, out) -> { out.escapeTime = -1.0; };
        FieldStack r = engine.sweep(new Viewport(-1.0, 1.0, -1.0, 1.0),
                new RenderSettings(2, 2, 2), k, 1);
        for (double[] row : r.escapeMap()) {
            for (double v : row) assertEquals(-1.0, v, 1e-9);
        }
    }

    @Test
    void ssaaBiasesMixedPixelToEscape() {
        IterationKernel k = (cRe, cIm, out) -> { out.escapeTime = cRe > 0.0 ? 10.0 : -1.0; };
        FieldStack r = engine.sweep(new Viewport(-1.0, 1.0, -0.5, 0.5),
                new RenderSettings(2, 1, 2), k, 1);
        // Left pixel: SSAA samples both at cRe < 0 → all in-set → -1.
        assertEquals(-1.0, r.escapeMap()[0][0], 1e-9);
        // Right pixel: SSAA samples both at cRe > 0 → both escape with v=10 → mean=10.
        assertEquals(10.0, r.escapeMap()[0][1], 1e-9);
    }

    @Test
    void deFieldAllocatedWhenKernelOptsIn() {
        IterationKernel k = new IterationKernel() {
            @Override public boolean supportsDistanceEstimate() { return true; }
            @Override public void sample(double cRe, double cIm, FieldsOut out) {
                out.escapeTime = 0.0;
                out.distanceEstimate = cRe + cIm;
            }
        };
        FieldStack r = engine.sweep(new Viewport(0.0, 1.0, 0.0, 1.0),
                new RenderSettings(2, 2), k, 1);
        assertNotNull(r.distanceMap(), "DE-supporting kernel must produce a distance map");
        // Top-left = (0, 1), bottom-right = (1, 0). Sum is 1 either way.
        assertEquals(1.0, r.distanceMap()[0][0], 1e-9);
        assertEquals(2.0, r.distanceMap()[0][1], 1e-9);
        assertEquals(0.0, r.distanceMap()[1][0], 1e-9);
        assertEquals(1.0, r.distanceMap()[1][1], 1e-9);
    }

    @Test
    void deAveragingSkipsNanSubsamples() {
        // Mixed kernel: half the SSAA grid emits NaN (in-set), half emits 5.0.
        IterationKernel k = new IterationKernel() {
            @Override public boolean supportsDistanceEstimate() { return true; }
            @Override public void sample(double cRe, double cIm, FieldsOut out) {
                if (cRe > 0.0) {
                    out.escapeTime = 1.0;
                    out.distanceEstimate = 5.0;
                } else {
                    out.escapeTime = -1.0;
                    out.distanceEstimate = Double.NaN;
                }
            }
        };
        FieldStack r = engine.sweep(new Viewport(-1.0, 1.0, -0.5, 0.5),
                new RenderSettings(2, 1, 2), k, 1);
        // Right pixel: both SSAA samples at cRe>0 → DE = mean(5, 5) = 5.
        assertEquals(5.0, r.distanceMap()[0][1], 1e-9);
        // Left pixel: all in-set → DE = NaN.
        assertTrue(Double.isNaN(r.distanceMap()[0][0]), "all-in-set DE must be NaN");
    }

    @Test
    void durationMsIsRecorded() {
        IterationKernel k = (cRe, cIm, out) -> { out.escapeTime = 1.0; };
        FieldStack r = engine.sweep(new Viewport(0.0, 1.0, 0.0, 1.0),
                new RenderSettings(8, 8), k, 1);
        assertTrue(r.durationMs() >= 0, "durationMs must be populated");
    }
}
