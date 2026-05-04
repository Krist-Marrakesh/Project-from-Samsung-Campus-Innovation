package com.fractalov.backend.service.color;

import com.fractalov.backend.dto.ColorMode;
import com.fractalov.backend.dto.Viewport;
import com.fractalov.backend.service.color.palettes.GrayscalePalette;
import com.fractalov.backend.service.render.RenderResult;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * DE colour mode coverage: NaN → black, presence of DE field maps through the
 * tanh ramp, and the fallback to LINEAR when the result has no DE map.
 */
class ColorizerDistanceTest {

    private final Colorizer colorizer = new Colorizer();
    private final Palette grayscale = new GrayscalePalette();

    @Test
    void deModeRendersInSetAsBlack() {
        double[][] escape = {{-1.0}};
        double[][] de = {{Double.NaN}};
        RenderResult r = new RenderResult(escape, de, 100, 0L);
        BufferedImage img = colorizer.colorize(
                r, new Viewport(0.0, 1.0, 0.0, 1.0), grayscale, ColorMode.DISTANCE_ESTIMATE);
        assertEquals(0xFF000000, img.getRGB(0, 0));
    }

    @Test
    void deModeMapsLargeDistanceToBrightShading() {
        // A finite, large DE value should land somewhere in the upper half of the ramp
        // because tanh(large) → 1.
        double[][] escape = {{1.0}};
        double[][] de = {{1e6}};
        RenderResult r = new RenderResult(escape, de, 100, 0L);
        BufferedImage img = colorizer.colorize(
                r, new Viewport(0.0, 1.0, 0.0, 1.0), grayscale, ColorMode.DISTANCE_ESTIMATE);
        int gray = img.getRGB(0, 0) & 0xFF;
        org.junit.jupiter.api.Assertions.assertTrue(gray > 200,
                "saturated DE should render bright, got " + gray);
    }

    @Test
    void deModeFallsBackToLinearWhenDistanceFieldMissing() {
        // Result has only escape map; requesting DE mode should silently degrade
        // to LINEAR rather than NPE.
        double[][] escape = {{50.0}};
        RenderResult r = new RenderResult(escape, null, 100, 0L);
        BufferedImage img = colorizer.colorize(
                r, new Viewport(0.0, 1.0, 0.0, 1.0), grayscale, ColorMode.DISTANCE_ESTIMATE);
        // Linear with grayscale: t = 50/100 = 0.5 → mid-gray, not in-set black.
        int rgb = img.getRGB(0, 0);
        assertNotEquals(0xFF000000, rgb,
                "DE→LINEAR fallback should produce mid-gray, not in-set black");
    }
}
