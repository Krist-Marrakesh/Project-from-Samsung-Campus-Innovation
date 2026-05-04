package com.fractalov.backend.service.render.kernel;

import com.fractalov.backend.dto.JuliaParams;
import com.fractalov.backend.service.render.FieldsOut;
import com.fractalov.backend.service.render.IterationKernel;

/**
 * Julia iteration kernel: {@code z_{n+1} = z_n² + c} with {@code c} fixed by
 * the recipe and {@code z_0 = (x, y)} drawn from the spatial grid.
 *
 * <h3>Distance estimation</h3>
 * Tracks {@code dz_n/dz_0} alongside the orbit:
 * {@code dz_{n+1}/dz_0 = 2·z_n·dz_n/dz_0}, {@code dz_0/dz_0 = 1}. The DE at
 * escape is {@code |z| · log|z| / |dz|} — same form as Mandelbrot but the
 * derivative is taken with respect to the spatial coordinate rather than the
 * parameter, which is the natural geometry for Julia sets.
 */
public final class JuliaKernel implements IterationKernel {

    private static final double LN2 = Math.log(2.0);

    private final int maxIter;
    private final double escR2;
    private final boolean smoothing;
    private final boolean computeDe;
    private final double cRe;
    private final double cIm;

    public JuliaKernel(JuliaParams p, boolean computeDe) {
        this.maxIter = p.maxIter();
        this.escR2 = p.escapeRadius() * p.escapeRadius();
        this.smoothing = p.smoothing();
        this.computeDe = computeDe;
        this.cRe = p.cRe();
        this.cIm = p.cIm();
    }

    @Override
    public boolean supportsDistanceEstimate() {
        return true;
    }

    @Override
    public void sample(double zRe0, double zIm0, FieldsOut out) {
        double zRe = zRe0;
        double zIm = zIm0;
        double dzRe = 1.0;
        double dzIm = 0.0;
        double zRe2 = zRe * zRe;
        double zIm2 = zIm * zIm;

        int iter = 0;
        while (iter < maxIter) {
            if (zRe2 + zIm2 > escR2) break;

            if (computeDe) {
                final double newDzRe = 2.0 * (zRe * dzRe - zIm * dzIm);
                final double newDzIm = 2.0 * (zRe * dzIm + zIm * dzRe);
                dzRe = newDzRe;
                dzIm = newDzIm;
            }

            final double zReNext = zRe2 - zIm2 + cRe;
            zIm = 2.0 * zRe * zIm + cIm;
            zRe = zReNext;
            zRe2 = zRe * zRe;
            zIm2 = zIm * zIm;
            iter++;
        }

        if (iter >= maxIter) {
            out.escapeTime = -1.0;
            out.distanceEstimate = Double.NaN;
            return;
        }

        if (smoothing) {
            final double mag2 = zRe2 + zIm2;
            final double logZn = 0.5 * Math.log(mag2);
            out.escapeTime = iter + 1.0 - Math.log(logZn / LN2) / LN2;
        } else {
            out.escapeTime = iter;
        }

        if (computeDe) {
            final double zMag = Math.sqrt(zRe2 + zIm2);
            final double dzMag = Math.sqrt(dzRe * dzRe + dzIm * dzIm);
            out.distanceEstimate = (dzMag > 0.0 && zMag > 0.0)
                    ? zMag * Math.log(zMag) / dzMag
                    : Double.NaN;
        }
    }
}
