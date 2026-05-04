package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.ColorSettings;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.MultibrotParams;
import com.fractalov.backend.dto.RenderSettings;
import com.fractalov.backend.dto.Viewport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MultibrotRendererTest {

    private final EscapeTimeEngine engine = new EscapeTimeEngine();
    private final MultibrotRenderer renderer = new MultibrotRenderer(engine);

    @Test
    void originIsInSetForAnyExponent() {
        for (int n : new int[]{2, 3, 5, 10}) {
            FractalRecipe recipe = new FractalRecipe(
                    new Viewport(-0.01, 0.01, -0.01, 0.01),
                    new RenderSettings(3, 3),
                    new ColorSettings("grayscale"),
                    new MultibrotParams(n, 200, 2.0, false)
            );
            RenderResult r = renderer.render(recipe);
            assertEquals(-1.0, r.escapeMap()[1][1], 1e-9, "origin should be in-set for N=" + n);
        }
    }

    @Test
    void exponentTwoMatchesMandelbrotSignature() {
        // N=2 recovers Mandelbrot. Both renderers over the same window should share
        // the same in-set pattern (match up to smoothing representation — we use !smoothing).
        Viewport vp = new Viewport(-2.0, 1.0, -1.2, 1.2);
        RenderSettings rs = new RenderSettings(24, 24);
        MandelbrotRenderer mandel = new MandelbrotRenderer(engine);
        double[][] multi = renderer.render(new FractalRecipe(
                vp, rs, new ColorSettings("grayscale"),
                new MultibrotParams(2, 200, 2.0, false))).escapeMap();
        double[][] mand = mandel.render(new FractalRecipe(
                vp, rs, new ColorSettings("grayscale"),
                new com.fractalov.backend.dto.MandelbrotParams(200, 2.0, false))).escapeMap();
        int agree = 0, total = 0;
        for (int y = 0; y < rs.heightPx(); y++) {
            for (int x = 0; x < rs.widthPx(); x++) {
                total++;
                boolean mIn = mand[y][x] < 0.0;
                boolean mu = multi[y][x] < 0.0;
                if (mIn == mu) agree++;
            }
        }
        // Multibrot uses polar form (slower numerical path), Mandelbrot closed form —
        // require near-identity on in-set classification.
        double ratio = (double) agree / total;
        assertTrue(ratio > 0.97, "N=2 multibrot vs mandelbrot membership ratio=" + ratio);
    }

    @Test
    void higherExponentHasRotationalSymmetry() {
        // N=3 set has 2-fold rotational symmetry (180°) in the real axis as well as
        // the standard complex-conjugate mirror. Check that escape-vs-in-set is
        // symmetric about the real axis.
        int n = 3;
        Viewport vp = new Viewport(-1.3, 1.3, -1.3, 1.3);
        RenderSettings rs = new RenderSettings(25, 25);
        double[][] m = renderer.render(new FractalRecipe(
                vp, rs, new ColorSettings("grayscale"),
                new MultibrotParams(n, 200, 2.0, false))).escapeMap();
        int h = 25;
        int w = 25;
        int match = 0, total = 0;
        for (int y = 0; y < h / 2; y++) {
            for (int x = 0; x < w; x++) {
                boolean a = m[y][x] < 0.0;
                boolean b = m[h - 1 - y][x] < 0.0;
                if (a == b) match++;
                total++;
            }
        }
        double ratio = (double) match / total;
        assertTrue(ratio > 0.95, "N=3 multibrot should be y-symmetric, got " + ratio);
    }
}
