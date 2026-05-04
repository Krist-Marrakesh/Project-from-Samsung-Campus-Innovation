package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.RenderSettings;
import com.fractalov.backend.dto.Viewport;
import org.springframework.stereotype.Component;

import java.util.stream.IntStream;

/**
 * Generalized escape-time renderer. Walks a viewport grid in parallel and
 * delegates per-point math to a supplied {@link IterationKernel}, then assembles
 * the per-pixel field outputs into a {@link FieldStack}.
 *
 * <p>This is the single computational substrate behind every fractal family in
 * the project. Mandelbrot, Julia, Burning Ship and Multibrot are all expressed
 * as {@link IterationKernel} implementations; the engine itself is family-agnostic.
 *
 * <h3>SSAA semantics</h3>
 * When {@link RenderSettings#effectiveSamplesPerAxis()} is {@code N > 1}, every
 * output pixel averages {@code N×N} jittered sub-samples placed at the centre of
 * each sub-cell. Two independent reductions run side-by-side:
 * <ul>
 *   <li><b>Escape-time field.</b> If <em>all</em> sub-samples are in-set
 *       (sentinel {@code -1.0}) the output is {@code -1.0}; otherwise escape
 *       sub-samples are arithmetic-meaned and in-set sub-samples are dropped.
 *       This biases mixed-boundary pixels toward escape coloring rather than
 *       blending darkness inward, which keeps the set's silhouette readable.</li>
 *   <li><b>Distance-estimate field.</b> Only finite (non-NaN) sub-sample DE
 *       values are averaged. If every sub-sample is in-set or NaN, the output
 *       DE is NaN.</li>
 * </ul>
 *
 * <h3>Allocation model</h3>
 * Each parallel row task allocates one {@link FieldsOut} scratch buffer and
 * reuses it across the row — the JIT can typically scalar-replace it. Output
 * arrays are pre-allocated once before the parallel walk.
 */
@Component
public class EscapeTimeEngine {

    /**
     * Run {@code kernel} over the (viewport, settings) grid and return the
     * combined field stack. {@code maxIter} is propagated into the result so
     * downstream consumers (colorizer, ML pipelines) know the iteration budget
     * without re-parsing the recipe.
     */
    public FieldStack sweep(Viewport vp, RenderSettings rs, IterationKernel kernel, int maxIter) {
        final int width = rs.widthPx();
        final int height = rs.heightPx();
        final int n = rs.effectiveSamplesPerAxis();
        final boolean wantDe = kernel.supportsDistanceEstimate();

        final double xMin = vp.xMin();
        final double xSpan = vp.xMax() - vp.xMin();
        final double yMax = vp.yMax();
        final double ySpan = vp.yMax() - vp.yMin();

        final double[][] escape = new double[height][width];
        final double[][] de = wantDe ? new double[height][width] : null;

        long start = System.nanoTime();
        if (n <= 1) {
            sweepSingle(width, height, xMin, xSpan, yMax, ySpan, kernel, escape, de);
        } else {
            sweepSsaa(width, height, n, xMin, xSpan, yMax, ySpan, kernel, escape, de);
        }
        long durationMs = (System.nanoTime() - start) / 1_000_000L;
        return new FieldStack(escape, de, maxIter, durationMs);
    }

    private static void sweepSingle(
            int width, int height,
            double xMin, double xSpan, double yMax, double ySpan,
            IterationKernel kernel,
            double[][] escape, double[][] de) {
        final double wDenom = width > 1 ? (width - 1) : 1.0;
        final double hDenom = height > 1 ? (height - 1) : 1.0;
        IntStream.range(0, height).parallel().forEach(py -> {
            final FieldsOut out = new FieldsOut();
            final double cIm = yMax - ySpan * py / hDenom;
            final double[] eRow = escape[py];
            final double[] dRow = de == null ? null : de[py];
            for (int px = 0; px < width; px++) {
                final double cRe = xMin + xSpan * px / wDenom;
                out.reset();
                kernel.sample(cRe, cIm, out);
                eRow[px] = out.escapeTime;
                if (dRow != null) dRow[px] = out.distanceEstimate;
            }
        });
    }

    private static void sweepSsaa(
            int width, int height, int n,
            double xMin, double xSpan, double yMax, double ySpan,
            IterationKernel kernel,
            double[][] escape, double[][] de) {
        final double pxW = xSpan / width;
        final double pxH = ySpan / height;
        final double step = 1.0 / n;
        final double halfStep = step * 0.5;

        IntStream.range(0, height).parallel().forEach(py -> {
            final FieldsOut out = new FieldsOut();
            final double[] eRow = escape[py];
            final double[] dRow = de == null ? null : de[py];
            for (int px = 0; px < width; px++) {
                double escapeSum = 0.0;
                int escapeCount = 0;
                double deSum = 0.0;
                int deCount = 0;
                for (int sy = 0; sy < n; sy++) {
                    final double cIm = yMax - pxH * (py + halfStep + sy * step);
                    for (int sx = 0; sx < n; sx++) {
                        final double cRe = xMin + pxW * (px + halfStep + sx * step);
                        out.reset();
                        kernel.sample(cRe, cIm, out);
                        if (out.escapeTime >= 0.0) {
                            escapeSum += out.escapeTime;
                            escapeCount++;
                        }
                        if (dRow != null && !Double.isNaN(out.distanceEstimate)) {
                            deSum += out.distanceEstimate;
                            deCount++;
                        }
                    }
                }
                eRow[px] = escapeCount == 0 ? -1.0 : escapeSum / escapeCount;
                if (dRow != null) {
                    dRow[px] = deCount == 0 ? Double.NaN : deSum / deCount;
                }
            }
        });
    }
}
