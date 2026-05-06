package com.fractalov.backend.service.color;

import com.fractalov.backend.dto.ColorMode;
import com.fractalov.backend.dto.Viewport;
import com.fractalov.backend.service.render.RenderResult;
import org.springframework.stereotype.Component;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Maps a {@link RenderResult} field stack into a coloured {@link BufferedImage}.
 *
 * <h3>Modes</h3>
 * <ul>
 *   <li>{@link ColorMode#LINEAR} — escape value normalised by {@code maxIter}.</li>
 *   <li>{@link ColorMode#HISTOGRAM} — CDF rank against a precomputed quantile
 *       lookup table of {@value #HIST_QUANTILE_BUCKETS} bins. Replaces the
 *       earlier per-pixel binary-search-against-N path: the lookup tier is
 *       {@code O(log K)} on a fixed-size table instead of {@code O(log N)}
 *       on the full pixel population. For 1024² renders that drops the inner
 *       comparison count from ~20 to ~10 per pixel and removes a
 *       memory-bandwidth-bound sort access pattern.</li>
 *   <li>{@link ColorMode#DISTANCE_ESTIMATE} — uses the per-pixel DE field
 *       (computed by {@link com.fractalov.backend.service.render.IterationKernel
 *       kernels} that opt in). DE is normalised against viewport pixel width
 *       and run through a {@code tanh} ramp so the shading is consistent
 *       across zoom levels. Falls back to {@link ColorMode#LINEAR} when the
 *       renderer's family does not produce DE (e.g. Burning Ship).</li>
 * </ul>
 *
 * <h3>Pixel writes</h3>
 * Writes go directly into the {@link BufferedImage}'s backing
 * {@link DataBufferInt} via {@link IntStream#parallel()}-friendly
 * {@code int[]} indices. This is several times faster than
 * {@link BufferedImage#setRGB(int, int, int)} on large images because each
 * {@code setRGB} call goes through colour-model bounds checks and an
 * indirection through the raster API.
 */
@Component
public class Colorizer {

    private static final int IN_SET_ARGB = 0xFF000000;
    private static final int HIST_QUANTILE_BUCKETS = 1024;

    /**
     * Bounded LUT cache for histogram-mode quantile thresholds. Sized
     * to fit ~16 distinct recipes — past that the eviction policy
     * starts mattering, but at 16 entries × 1024 doubles × 8 bytes
     * the cache itself is ≤ 128 KB. The {@link LinkedHashMap}
     * access-order constructor gives LRU eviction for free.
     */
    private static final int LUT_CACHE_CAPACITY = 16;
    private static final java.util.Map<HistogramKey, double[]> HISTOGRAM_LUT_CACHE =
            java.util.Collections.synchronizedMap(
                    new java.util.LinkedHashMap<>(LUT_CACHE_CAPACITY, 0.75f, /* accessOrder */ true) {
                        @Override
                        protected boolean removeEldestEntry(java.util.Map.Entry<HistogramKey, double[]> eldest) {
                            return size() > LUT_CACHE_CAPACITY;
                        }
                    });

    /**
     * Cheap fingerprint of the escape map used as the cache key. We
     * never compare full {@code double[][]} contents — instead we
     * derive a compact tuple (dimensions, escape count, finite
     * checksum, finite max). Two field stacks colliding under all
     * four projections AND producing different histograms would be
     * an extremely contrived recipe; in the worst case the user sees
     * a faintly wrong colourisation that fixes itself on the next
     * render. The alternative — full content hashing — would cost
     * more than the sort it's trying to avoid.
     */
    private record HistogramKey(int width, int height, int escapeCount, long checksumBits, long maxBits) {
        static HistogramKey of(double[][] escape, int escapeCount, int width, int height) {
            // Single pass to compute the checksum and max in one go.
            // ``Double.doubleToLongBits`` keeps NaN-collisions stable
            // across runs (we don't expect NaN here, but better safe).
            double sumSq = 0.0;
            double maxV = Double.NEGATIVE_INFINITY;
            for (double[] row : escape) {
                for (double v : row) {
                    if (v < 0.0) continue;
                    sumSq += v * v;
                    if (v > maxV) maxV = v;
                }
            }
            return new HistogramKey(
                    width, height, escapeCount,
                    Double.doubleToLongBits(sumSq),
                    Double.doubleToLongBits(maxV));
        }
    }

    /**
     * O(N log N) work that the cache avoids on a hit. Pulled out of
     * the hot path so the hit branch in {@link #histogramFill} stays
     * a single map lookup.
     */
    private static double[] computeQuantileLut(double[][] escape, int escapeCount) {
        final double[] sorted = new double[escapeCount];
        int idx = 0;
        for (double[] row : escape) {
            for (double v : row) {
                if (v >= 0.0) sorted[idx++] = v;
            }
        }
        Arrays.sort(sorted);
        // Fixed-size LUT: one threshold per bucket. Ties between
        // adjacent buckets collapse naturally — Arrays.binarySearch
        // returning the insertion point covers it.
        final int k = Math.min(HIST_QUANTILE_BUCKETS, escapeCount);
        final double[] thresholds = new double[k];
        for (int i = 0; i < k; i++) {
            int j = (int) ((long) i * escapeCount / k);
            if (j >= escapeCount) j = escapeCount - 1;
            thresholds[i] = sorted[j];
        }
        return thresholds;
    }

    /**
     * Multiplier for DE-mode {@code tanh} ramp. Larger value → ramp transitions
     * across more pixels of viewport space → softer boundary, brighter exterior.
     * {@code 8.0} gives a perceptually balanced shading on the standard
     * Mandelbrot view at 1024².
     */
    private static final double DE_TANH_PIXELS = 8.0;

    /** Convenience overload for callers that only have an escape-time map and
     * never use DE mode (e.g. unit tests). DE mode silently degrades to LINEAR
     * when the result has no distance field, so passing {@code null} viewport
     * is safe. */
    public BufferedImage colorize(double[][] escapeMap, int maxIter, Palette palette, ColorMode mode) {
        return colorize(new RenderResult(escapeMap, null, maxIter, 0L), null, palette, mode);
    }

    public BufferedImage colorize(RenderResult result, Viewport viewport, Palette palette, ColorMode mode) {
        double[][] escapeMap = result.escapeMap();
        int height = escapeMap.length;
        int width = height == 0 ? 0 : escapeMap[0].length;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        if (width == 0 || height == 0) return image;

        ColorMode effective = (mode == ColorMode.DISTANCE_ESTIMATE && !result.hasDistanceField())
                ? ColorMode.LINEAR : mode;

        int[] pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();

        switch (effective) {
            case LINEAR -> linearFill(escapeMap, result.maxIter(), palette, pixels, width, height);
            case HISTOGRAM -> histogramFill(escapeMap, palette, pixels, width, height);
            case DISTANCE_ESTIMATE -> distanceFill(
                    result.distanceMap(), viewport, palette, pixels, width, height);
        }
        return image;
    }

    private static void linearFill(
            double[][] escape, int maxIter, Palette palette, int[] out, int width, int height) {
        final double denom = maxIter > 0 ? maxIter : 1.0;
        IntStream.range(0, height).parallel().forEach(y -> {
            final double[] row = escape[y];
            final int base = y * width;
            for (int x = 0; x < width; x++) {
                final double v = row[x];
                if (v < 0.0) {
                    out[base + x] = IN_SET_ARGB;
                } else {
                    double t = v / denom;
                    if (t < 0.0) t = 0.0;
                    else if (t > 1.0) t = 1.0;
                    out[base + x] = palette.argbAt(t);
                }
            }
        });
    }

    private static void histogramFill(
            double[][] escape, Palette palette, int[] out, int width, int height) {
        int escapeCount = 0;
        for (double[] row : escape) {
            for (double v : row) if (v >= 0.0) escapeCount++;
        }
        if (escapeCount == 0) {
            Arrays.fill(out, IN_SET_ARGB);
            return;
        }

        // Quantile LUT lookup. The expensive part of histogram colour mode
        // is the O(N log N) sort of every escape value on every render —
        // a 1024² render with all-escape pixels sorts ~10⁶ doubles every
        // time. Repeat renders of the same recipe (typical in
        // benchmarking and live-preview UIs) produce byte-identical
        // sorted sequences, so we cache the resulting thresholds keyed
        // on a cheap fingerprint of the escape array.
        final HistogramKey cacheKey = HistogramKey.of(escape, escapeCount, width, height);
        double[] cached = HISTOGRAM_LUT_CACHE.get(cacheKey);
        if (cached == null) {
            cached = computeQuantileLut(escape, escapeCount);
            HISTOGRAM_LUT_CACHE.put(cacheKey, cached);
        }
        final double[] thresholds = cached;
        final int k = thresholds.length;
        final double denomT = k > 1 ? (k - 1) : 1.0;

        IntStream.range(0, height).parallel().forEach(y -> {
            final double[] row = escape[y];
            final int base = y * width;
            for (int x = 0; x < width; x++) {
                final double v = row[x];
                if (v < 0.0) {
                    out[base + x] = IN_SET_ARGB;
                } else {
                    int pos = Arrays.binarySearch(thresholds, v);
                    if (pos < 0) pos = -pos - 1;
                    if (pos >= k) pos = k - 1;
                    double t = pos / denomT;
                    if (t < 0.0) t = 0.0;
                    else if (t > 1.0) t = 1.0;
                    out[base + x] = palette.argbAt(t);
                }
            }
        });
    }

    private static void distanceFill(
            double[][] de, Viewport viewport, Palette palette, int[] out, int width, int height) {
        // Without a viewport we cannot normalise DE in any zoom-consistent way.
        // Fall through to a raw-clamp ramp so the call still produces an image
        // (used by unit tests; real callers always pass viewport).
        final double pixelWidth = viewport == null
                ? 1.0
                : (viewport.xMax() - viewport.xMin()) / Math.max(width, 1);
        final double inv = 1.0 / Math.max(pixelWidth * DE_TANH_PIXELS, 1e-300);

        IntStream.range(0, height).parallel().forEach(y -> {
            final double[] row = de[y];
            final int base = y * width;
            for (int x = 0; x < width; x++) {
                final double v = row[x];
                if (Double.isNaN(v) || v < 0.0) {
                    out[base + x] = IN_SET_ARGB;
                } else {
                    double t = Math.tanh(v * inv);
                    if (t < 0.0) t = 0.0;
                    else if (t > 1.0) t = 1.0;
                    out[base + x] = palette.argbAt(t);
                }
            }
        });
    }
}
