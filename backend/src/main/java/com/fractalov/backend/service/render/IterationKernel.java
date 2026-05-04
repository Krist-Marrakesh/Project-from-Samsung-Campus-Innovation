package com.fractalov.backend.service.render;

/**
 * Stateless, thread-safe per-point iteration kernel — the abstract operation
 * a fractal family contributes to the {@link EscapeTimeEngine}. Replaces the
 * older single-output {@code PixelKernel} with a multi-field protocol so the
 * engine can compute escape-time, distance estimate, and (later) other derived
 * fields in one pass over the iteration loop.
 *
 * <p>Invariants:
 * <ul>
 *   <li>Implementations MUST populate {@link FieldsOut#escapeTime} on every call.</li>
 *   <li>Implementations that report {@link #supportsDistanceEstimate()} {@code = true}
 *       MUST populate {@link FieldsOut#distanceEstimate} (NaN if in-set), or leave
 *       it as the {@link FieldsOut#reset()} default ({@link Double#NaN}) for in-set.
 *       Kernels that do not support DE leave {@code distanceEstimate} as NaN.</li>
 *   <li>{@link #sample} is called concurrently from parallel streams. Implementations
 *       must be stateless or at least lock-free with respect to the per-call
 *       {@link FieldsOut} buffer (which the engine owns and never shares between
 *       threads).</li>
 * </ul>
 *
 * <p>The kernel captures recipe-specific constants (maxIter, escapeRadius²,
 * smoothing flag, family-specific params) at construction time. The engine
 * captures viewport / sampling concerns. This separation is deliberate: the
 * same kernel can be reused across SSAA sub-samples without rebuilding, and
 * the engine can share its parallel walk across all four families.
 */
public interface IterationKernel {

    /**
     * Run the family's iteration on a single complex point and write derived
     * fields into {@code out}. Must always populate {@link FieldsOut#escapeTime}.
     */
    void sample(double cRe, double cIm, FieldsOut out);

    /**
     * Whether this kernel populates {@link FieldsOut#distanceEstimate}. Used by
     * the engine to decide whether to allocate the parallel DE field array, and
     * by the colorizer to decide whether {@code DISTANCE_ESTIMATE} mode is
     * available for this render. Default {@code false} — kernels opt in.
     */
    default boolean supportsDistanceEstimate() {
        return false;
    }
}
