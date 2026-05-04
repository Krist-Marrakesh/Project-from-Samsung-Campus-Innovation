package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.FractalType;
import com.fractalov.backend.dto.MandelbrotParams;
import com.fractalov.backend.dto.RenderSettings;
import com.fractalov.backend.dto.Viewport;
import org.springframework.stereotype.Component;

import java.util.stream.IntStream;

/**
 * Optimised scalar Mandelbrot renderer — Stage 9 research artefact.
 *
 * <p>The naive {@link MandelbrotRenderer} is already vectorised in spirit
 * (parallelStream + flat double-loop), so the cheapest measurable speedup is
 * <b>algorithmic</b>, not microarchitectural. Two well-known tricks:
 *
 * <ol>
 *   <li><b>Cardioid + period-2 bulb early bail.</b> Two simple analytic
 *       inequalities tell us a point is in-set <em>before</em> running any
 *       iterations: the main cardioid covers ~40% of the set's area, the
 *       period-2 bulb another ~10%. Skipping the iteration loop entirely
 *       for those points removes a huge amount of work on any view that
 *       includes the body of the set.</li>
 *   <li><b>Periodicity check.</b> If {@code z_n == z_{n-k}} for some small
 *       {@code k}, the orbit is periodic and the point is in-set. Cheap to
 *       implement (snapshot every 20 iterations, compare to current) and
 *       catches the deep-zoom cases the cardioid test misses.</li>
 * </ol>
 *
 * <p><b>Why not the Java 21 Vector API.</b> The original Stage 9 plan was a
 * SIMD renderer using {@code jdk.incubator.vector.DoubleVector}. On Apple
 * Silicon (NEON, 2-lane double) we measured a ~100× <em>slowdown</em>
 * instead — Java 21 lacks vectorised intrinsics for several double
 * operations on ARM, so {@code zRe.mul(zRe)} falls back to a generic boxing
 * path. This is itself documented in the Stage 9 research report. Switching
 * to algorithmic optimisation gave a measurable improvement on the same
 * test machine.
 *
 * <p>This class deliberately is <em>not</em> a {@link FractalRenderer} bean
 * to avoid clashing with the production {@link MandelbrotRenderer}'s
 * {@code supports() = MANDELBROT}; the {@code /bench/compare} endpoint
 * picks it up directly via Spring DI by name.
 */
@Component
public class MandelbrotVectorRenderer {

    private static final double LN2 = Math.log(2.0);
    private static final int PERIODICITY_INTERVAL = 20;
    private static final double PERIODICITY_EPSILON = 1e-12;

    public FractalType supports() {
        return FractalType.MANDELBROT;
    }

    /** Cosmetic: the {@code /bench/compare} response carries this so the report
     * can label rows. We keep the name "lane count" but it's really just 1 here
     * — the optimisation is algorithmic, not SIMD. */
    public int laneCount() {
        return 1;
    }

    public RenderResult render(FractalRecipe recipe) {
        if (!(recipe.params() instanceof MandelbrotParams p)) {
            throw new RenderException("MandelbrotVectorRenderer requires MandelbrotParams");
        }
        Viewport vp = recipe.viewport();
        RenderSettings rs = recipe.renderSettings();

        final int width = rs.widthPx();
        final int height = rs.heightPx();
        final int maxIter = p.maxIter();
        final double escR2 = p.escapeRadius() * p.escapeRadius();
        final boolean smoothing = p.smoothing();

        final double xMin = vp.xMin();
        final double xSpan = vp.xMax() - vp.xMin();
        final double yMax = vp.yMax();
        final double ySpan = vp.yMax() - vp.yMin();
        final double wDenom = width > 1 ? (width - 1) : 1.0;
        final double hDenom = height > 1 ? (height - 1) : 1.0;

        final double[][] escapeMap = new double[height][width];

        long start = System.nanoTime();

        IntStream.range(0, height).parallel().forEach(py -> {
            final double cIm = yMax - ySpan * py / hDenom;
            final double[] row = escapeMap[py];
            for (int px = 0; px < width; px++) {
                final double cRe = xMin + xSpan * px / wDenom;

                if (insideMainCardioid(cRe, cIm) || insidePeriod2Bulb(cRe, cIm)) {
                    row[px] = -1.0;
                    continue;
                }

                row[px] = iterate(cRe, cIm, maxIter, escR2, smoothing);
            }
        });

        long durationMs = (System.nanoTime() - start) / 1_000_000L;
        return new RenderResult(escapeMap, maxIter, durationMs);
    }

    /** Main cardioid: {@code |1 - sqrt(1 - 4c)| < 1}, expanded into the closed
     * form {@code q*(q + (cRe - 0.25)) < 0.25 * cIm²} where
     * {@code q = (cRe - 0.25)² + cIm²}. */
    private static boolean insideMainCardioid(double cRe, double cIm) {
        final double dx = cRe - 0.25;
        final double q = dx * dx + cIm * cIm;
        return q * (q + dx) < 0.25 * cIm * cIm;
    }

    /** Period-2 bulb centred at (-1, 0) with radius 1/4. */
    private static boolean insidePeriod2Bulb(double cRe, double cIm) {
        final double dx = cRe + 1.0;
        return dx * dx + cIm * cIm < 0.0625; // (1/4)²
    }

    /** Iteration loop with periodicity check. Snapshots {@code (zRe, zIm)}
     * every {@code PERIODICITY_INTERVAL} iterations; if the orbit revisits
     * the snapshot, declares in-set early. */
    private static double iterate(double cRe, double cIm, int maxIter, double escR2, boolean smoothing) {
        double zRe = 0.0, zIm = 0.0;
        double zRe2 = 0.0, zIm2 = 0.0;
        double snapRe = 0.0, snapIm = 0.0;

        for (int iter = 0; iter < maxIter; iter++) {
            zRe2 = zRe * zRe;
            zIm2 = zIm * zIm;
            if (zRe2 + zIm2 > escR2) {
                if (!smoothing) return iter;
                final double mag2 = zRe2 + zIm2;
                final double logZn = 0.5 * Math.log(mag2);
                return iter + 1.0 - Math.log(logZn / LN2) / LN2;
            }
            final double zReNext = zRe2 - zIm2 + cRe;
            zIm = 2.0 * zRe * zIm + cIm;
            zRe = zReNext;

            // Periodicity check — every PERIODICITY_INTERVAL iterations,
            // snapshot z and look for a future revisit.
            if ((iter & (PERIODICITY_INTERVAL - 1)) == PERIODICITY_INTERVAL - 1) {
                if (Math.abs(zRe - snapRe) < PERIODICITY_EPSILON
                        && Math.abs(zIm - snapIm) < PERIODICITY_EPSILON) {
                    return -1.0;
                }
                snapRe = zRe;
                snapIm = zIm;
            }
        }
        return -1.0;
    }
}
