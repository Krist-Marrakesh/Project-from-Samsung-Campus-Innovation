package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.ColorSettings;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.MandelbrotParams;
import com.fractalov.backend.dto.RenderSettings;
import com.fractalov.backend.dto.Viewport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Vector API renderer is a research artefact, but the whole point is that
 * it produces the <em>same</em> picture as the scalar version (otherwise the
 * benchmarks would be measuring a faster bug). These tests assert two things:
 *
 * <ol>
 *   <li><b>In-set classification matches exactly.</b> Whether a pixel is
 *       in-set or escapes is a discrete decision — the two implementations
 *       must agree on every pixel.</li>
 *   <li><b>Continuous escape values agree to ~1e-6.</b> Smoothing's
 *       {@code log/log2} chain is non-associative, so SIMD reordering can
 *       drift by a few ULPs; the tolerance leaves room for that without
 *       letting genuine algorithmic errors slip through.</li>
 * </ol>
 */
class MandelbrotVectorParityTest {

    private final EscapeTimeEngine engine = new EscapeTimeEngine();
    private final MandelbrotRenderer scalar = new MandelbrotRenderer(engine);
    private final MandelbrotVectorRenderer vector = new MandelbrotVectorRenderer();

    private FractalRecipe recipe(int width, int height, int maxIter, boolean smoothing) {
        return new FractalRecipe(
                new Viewport(-2.0, 1.0, -1.2, 1.2),
                new RenderSettings(width, height),
                new ColorSettings("grayscale"),
                new MandelbrotParams(maxIter, 2.0, smoothing)
        );
    }

    @Test
    void inSetClassificationIsIdentical_smoothingOff() {
        FractalRecipe r = recipe(64, 64, 200, false);
        double[][] s = scalar.render(r).escapeMap();
        double[][] v = vector.render(r).escapeMap();
        assertSentinelsAgree(s, v);
    }

    @Test
    void inSetClassificationIsIdentical_smoothingOn() {
        FractalRecipe r = recipe(64, 64, 200, true);
        double[][] s = scalar.render(r).escapeMap();
        double[][] v = vector.render(r).escapeMap();
        assertSentinelsAgree(s, v);
    }

    @Test
    void smoothEscapeValuesAgreeToTinyTolerance() {
        FractalRecipe r = recipe(64, 64, 200, true);
        double[][] s = scalar.render(r).escapeMap();
        double[][] v = vector.render(r).escapeMap();
        double maxDiff = 0.0;
        int compared = 0;
        for (int y = 0; y < s.length; y++) {
            for (int x = 0; x < s[y].length; x++) {
                if (s[y][x] < 0.0) continue; // in-set already covered above
                double diff = Math.abs(s[y][x] - v[y][x]);
                if (diff > maxDiff) maxDiff = diff;
                compared++;
            }
        }
        assertTrue(compared > 0, "no escape pixels — recipe degenerated");
        assertTrue(maxDiff < 1e-5,
                "smoothed escape values drifted: max |scalar - vector| = " + maxDiff);
    }

    @Test
    void widthNotMultipleOfLaneCountStillWorks() {
        // Force the tail-handling path: width = 17 won't divide most SIMD lane
        // widths, so the last batch must be partially masked.
        FractalRecipe r = recipe(17, 13, 200, false);
        double[][] s = scalar.render(r).escapeMap();
        double[][] v = vector.render(r).escapeMap();
        assertEquals(13, v.length);
        assertEquals(17, v[0].length);
        assertSentinelsAgree(s, v);
    }

    private void assertSentinelsAgree(double[][] s, double[][] v) {
        for (int y = 0; y < s.length; y++) {
            for (int x = 0; x < s[y].length; x++) {
                boolean sIn = s[y][x] < 0.0;
                boolean vIn = v[y][x] < 0.0;
                assertEquals(sIn, vIn,
                        "in-set mismatch at (" + x + "," + y + "): scalar=" + s[y][x] + " vector=" + v[y][x]);
            }
        }
    }
}
