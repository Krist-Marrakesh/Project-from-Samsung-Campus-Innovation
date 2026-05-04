package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.ColorSettings;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.MandelbrotParams;
import com.fractalov.backend.dto.RenderSettings;
import com.fractalov.backend.dto.Viewport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MandelbrotRendererTest {

    private final MandelbrotRenderer renderer = new MandelbrotRenderer(new EscapeTimeEngine());

    @Test
    void centerIsInsideSet() {
        // 3x3 grid centered on (0,0) — the center pixel samples c=0+0i, which is in the set.
        FractalRecipe recipe = new FractalRecipe(
                new Viewport(-0.1, 0.1, -0.1, 0.1),
                new RenderSettings(3, 3),
                new ColorSettings("grayscale"),
                new MandelbrotParams(200, 2.0, false)
        );
        RenderResult r = renderer.render(recipe);
        assertEquals(-1.0, r.escapeMap()[1][1], 1e-9, "center pixel should be in-set (-1.0 sentinel)");
    }

    @Test
    void farPointEscapesQuickly() {
        // c = 4+4i is way outside the set, should escape on iter 0 or 1.
        FractalRecipe recipe = new FractalRecipe(
                new Viewport(4.0, 4.0001, 4.0, 4.0001),
                new RenderSettings(1, 1),
                new ColorSettings("grayscale"),
                new MandelbrotParams(200, 2.0, false)
        );
        RenderResult r = renderer.render(recipe);
        double v = r.escapeMap()[0][0];
        assertTrue(v >= 0.0 && v < 5.0, "far point should escape quickly, got " + v);
    }

    @Test
    void smoothingProducesFractionalValues() {
        FractalRecipe recipe = new FractalRecipe(
                new Viewport(-2.0, 1.0, -1.2, 1.2),
                new RenderSettings(16, 16),
                new ColorSettings("grayscale"),
                new MandelbrotParams(200, 2.0, true)
        );
        RenderResult r = renderer.render(recipe);
        boolean anyFractional = false;
        for (double[] row : r.escapeMap()) {
            for (double v : row) {
                if (v > 0.0 && Math.abs(v - Math.rint(v)) > 1e-6) {
                    anyFractional = true;
                    break;
                }
            }
            if (anyFractional) break;
        }
        assertTrue(anyFractional, "smoothing should produce at least one fractional escape value");
    }
}
