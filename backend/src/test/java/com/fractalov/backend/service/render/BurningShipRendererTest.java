package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.BurningShipParams;
import com.fractalov.backend.dto.ColorSettings;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.RenderSettings;
import com.fractalov.backend.dto.Viewport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BurningShipRendererTest {

    private final BurningShipRenderer renderer = new BurningShipRenderer(new EscapeTimeEngine());

    @Test
    void originIsInSet() {
        // c = 0 → iteration fixed at zero forever.
        FractalRecipe recipe = new FractalRecipe(
                new Viewport(-0.01, 0.01, -0.01, 0.01),
                new RenderSettings(3, 3),
                new ColorSettings("grayscale"),
                new BurningShipParams(200, 2.0, false)
        );
        RenderResult r = renderer.render(recipe);
        assertEquals(-1.0, r.escapeMap()[1][1], 1e-9);
    }

    @Test
    void farPointEscapes() {
        FractalRecipe recipe = new FractalRecipe(
                new Viewport(5.0, 5.0001, 5.0, 5.0001),
                new RenderSettings(1, 1),
                new ColorSettings("grayscale"),
                new BurningShipParams(200, 2.0, false)
        );
        RenderResult r = renderer.render(recipe);
        double v = r.escapeMap()[0][0];
        assertTrue(v >= 0.0 && v < 5.0, "escape expected, got " + v);
    }

    @Test
    void classicWindowHasBothInSetAndEscape() {
        FractalRecipe recipe = new FractalRecipe(
                new Viewport(-2.0, 1.5, -2.0, 1.0),
                new RenderSettings(24, 24),
                new ColorSettings("grayscale"),
                new BurningShipParams(200, 2.0, false)
        );
        RenderResult r = renderer.render(recipe);
        boolean hasInSet = false, hasEscape = false;
        for (double[] row : r.escapeMap()) {
            for (double v : row) {
                if (v < 0.0) hasInSet = true;
                else hasEscape = true;
            }
        }
        assertTrue(hasInSet, "expected in-set pixels");
        assertTrue(hasEscape, "expected escape pixels");
    }
}
