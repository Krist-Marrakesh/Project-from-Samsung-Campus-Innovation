package com.fractalov.backend.service.render;

/**
 * Mutable per-pixel scratch buffer that an {@link IterationKernel} fills with
 * the scalar fields it derives from a single point. One buffer is reused across
 * an entire pixel row (or SSAA cluster) to avoid per-pixel allocation in the
 * hot loop — the engine is responsible for resetting between samples.
 *
 * <p>Conventions:
 * <ul>
 *   <li>{@link #escapeTime} carries either {@code -1.0} (in-set sentinel),
 *       an integer iteration count (smoothing off), or a continuous value
 *       (smoothing on). Always populated by the kernel.</li>
 *   <li>{@link #distanceEstimate} is a non-negative scalar in viewport units —
 *       the rough distance from this point to the boundary of the set. It is
 *       {@link Double#NaN} when the point is in-set, when the kernel does not
 *       support DE, or when the kernel was asked to skip the derivative track.</li>
 * </ul>
 */
public final class FieldsOut {

    public double escapeTime;
    public double distanceEstimate;

    public void reset() {
        escapeTime = 0.0;
        distanceEstimate = Double.NaN;
    }
}
