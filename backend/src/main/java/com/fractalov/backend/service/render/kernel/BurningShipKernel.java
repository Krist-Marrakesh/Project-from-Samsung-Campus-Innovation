package com.fractalov.backend.service.render.kernel;

import com.fractalov.backend.dto.BurningShipParams;
import com.fractalov.backend.service.render.FieldsOut;
import com.fractalov.backend.service.render.IterationKernel;

/**
 * Burning Ship iteration kernel:
 * {@code z_{n+1} = (|Re z_n| + i·|Im z_n|)² + c}, {@code z_0 = 0}.
 *
 * <p>The absolute-value step on each axis breaks the holomorphic structure of
 * the Mandelbrot map, which means the standard distance estimator
 * {@code |z| · log|z| / |dz/dc|} does <em>not</em> apply — the derivative
 * {@code dz/dc} is undefined at every fold ({@code Re z = 0} or {@code Im z = 0}).
 * Reflecting this honestly, the kernel reports {@code supportsDistanceEstimate()
 * = false}; downstream colorizers must fall back to escape-time mode for
 * Burning Ship renders.
 */
public final class BurningShipKernel implements IterationKernel {

    private static final double LN2 = Math.log(2.0);

    private final int maxIter;
    private final double escR2;
    private final boolean smoothing;

    public BurningShipKernel(BurningShipParams p) {
        this.maxIter = p.maxIter();
        this.escR2 = p.escapeRadius() * p.escapeRadius();
        this.smoothing = p.smoothing();
    }

    @Override
    public boolean supportsDistanceEstimate() {
        return false;
    }

    @Override
    public void sample(double cRe, double cIm, FieldsOut out) {
        double zRe = 0.0, zIm = 0.0;
        double zRe2 = 0.0, zIm2 = 0.0;

        int iter = 0;
        while (iter < maxIter) {
            zRe2 = zRe * zRe;
            zIm2 = zIm * zIm;
            if (zRe2 + zIm2 > escR2) break;
            // (|zRe| + i|zIm|)² = zRe² − zIm² + i · 2|zRe·zIm|
            final double zReNext = zRe2 - zIm2 + cRe;
            zIm = 2.0 * Math.abs(zRe * zIm) + cIm;
            zRe = zReNext;
            iter++;
        }

        if (iter >= maxIter) {
            out.escapeTime = -1.0;
            return;
        }

        if (smoothing) {
            final double mag2 = zRe * zRe + zIm * zIm;
            final double logZn = 0.5 * Math.log(mag2);
            out.escapeTime = iter + 1.0 - Math.log(logZn / LN2) / LN2;
        } else {
            out.escapeTime = iter;
        }
    }
}
