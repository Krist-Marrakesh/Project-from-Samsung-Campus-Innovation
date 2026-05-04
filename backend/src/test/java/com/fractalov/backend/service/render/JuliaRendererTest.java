package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.ColorSettings;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.JuliaParams;
import com.fractalov.backend.dto.RenderSettings;
import com.fractalov.backend.dto.Viewport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JuliaRendererTest {

    private final JuliaRenderer renderer = new JuliaRenderer(new EscapeTimeEngine());

    @Test
    void originInSetWhenCIsZero() {
        // With c=0 the iteration z -> z^2 trivially keeps |z|<1 for |z_0|<1.
        // Center pixel samples z_0=0, which is a fixed point of z^2, so it must be in-set.
        FractalRecipe recipe = new FractalRecipe(
                new Viewport(-0.5, 0.5, -0.5, 0.5),
                new RenderSettings(3, 3),
                new ColorSettings("grayscale"),
                new JuliaParams(0.0, 0.0, 200, 2.0, false)
        );
        RenderResult r = renderer.render(recipe);
        assertEquals(-1.0, r.escapeMap()[1][1], 1e-9, "origin must be in-set when c=0");
    }

    @Test
    void classicJuliaProducesMixedMap() {
        // c=-0.7+0.27015i is a classic Julia set — a rendered viewport spanning its
        // characteristic region must have both escape and in-set pixels.
        FractalRecipe recipe = new FractalRecipe(
                new Viewport(-1.5, 1.5, -1.5, 1.5),
                new RenderSettings(32, 32),
                new ColorSettings("grayscale"),
                new JuliaParams(-0.7, 0.27015, 200, 2.0, false)
        );
        RenderResult r = renderer.render(recipe);
        boolean hasInSet = false, hasEscape = false;
        for (double[] row : r.escapeMap()) {
            for (double v : row) {
                if (v < 0.0) hasInSet = true;
                else hasEscape = true;
            }
        }
        assertTrue(hasInSet, "expected at least one in-set pixel");
        assertTrue(hasEscape, "expected at least one escape pixel");
    }

    @Test
    void farPointEscapes() {
        FractalRecipe recipe = new FractalRecipe(
                new Viewport(5.0, 5.0001, 5.0, 5.0001),
                new RenderSettings(1, 1),
                new ColorSettings("grayscale"),
                new JuliaParams(-0.7, 0.27015, 200, 2.0, false)
        );
        RenderResult r = renderer.render(recipe);
        double v = r.escapeMap()[0][0];
        assertTrue(v >= 0.0 && v < 3.0, "far point should escape immediately, got " + v);
    }
}
