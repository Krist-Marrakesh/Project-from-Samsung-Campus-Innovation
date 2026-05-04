package com.fractalov.backend.service.render;

/**
 * Result of an {@link EscapeTimeEngine#sweep} call: a fixed-shape multi-channel
 * raster with one row of {@link IterationKernel#sample} output per scanline.
 *
 * <p>{@link #escapeMap} is always present. {@link #distanceMap} is non-null only
 * when the kernel supports distance estimation; downstream colorizers must check.
 */
public record FieldStack(
        double[][] escapeMap,
        double[][] distanceMap,
        int maxIter,
        long durationMs
) {

    public boolean hasDistanceField() {
        return distanceMap != null;
    }
}
