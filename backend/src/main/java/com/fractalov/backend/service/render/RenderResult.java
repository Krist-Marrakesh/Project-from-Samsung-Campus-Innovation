package com.fractalov.backend.service.render;

/**
 * Result of a single render kernel invocation. Carries the per-pixel field
 * stack — {@code escapeMap} is always present, {@code distanceMap} is non-null
 * only when the underlying {@link IterationKernel} supports distance estimation
 * <em>and</em> the recipe asked for it via the colour mode.
 *
 * <p>The legacy three-arg constructor preserves call sites from before the
 * field-stack refactor; new code should pass the DE map explicitly (or
 * {@code null} when not computed).
 */
public record RenderResult(
        double[][] escapeMap,
        double[][] distanceMap,
        int maxIter,
        long durationMs
) {

    public RenderResult(double[][] escapeMap, int maxIter, long durationMs) {
        this(escapeMap, null, maxIter, durationMs);
    }

    public boolean hasDistanceField() {
        return distanceMap != null;
    }

    public static RenderResult fromFieldStack(FieldStack stack) {
        return new RenderResult(stack.escapeMap(), stack.distanceMap(),
                stack.maxIter(), stack.durationMs());
    }
}
