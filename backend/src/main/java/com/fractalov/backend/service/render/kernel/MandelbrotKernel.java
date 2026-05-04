package com.fractalov.backend.service.render.kernel;

import com.fractalov.backend.dto.MandelbrotParams;
import com.fractalov.backend.service.render.FieldsOut;
import com.fractalov.backend.service.render.IterationKernel;

/**
 * Mandelbrot iteration kernel: {@code z_{n+1} = z_n² + c}, {@code z_0 = 0}.
 *
 * <h3>In-set short-circuits (always on)</h3>
 * Before any iteration the kernel checks two analytic in-set regions:
 * <ul>
 *   <li><b>Main cardioid</b> — closed-form inequality covering ~40% of the set's
 *       area. A point inside is guaranteed in-set, no iteration needed.</li>
 *   <li><b>Period-2 bulb</b> centred at {@code (-1, 0)} with radius {@code 1/4}.
 *       Another ~10% of the set's area, period-2 attractor.</li>
 * </ul>
 * Combined, this skips the iteration loop for ~50% of points on a whole-set
 * view. Originally introduced as Stage 9 research artefact in the Vector kernel,
 * now adopted into the production path because the speedup is real and the
 * branches are cheap (one squared-distance comparison each).
 *
 * <h3>Distance estimation</h3>
 * When {@code computeDe} is true the iteration also tracks the derivative
 * {@code dz/dc}, recurrence {@code dz_{n+1} = 2·z_n·dz_n + 1} with
 * {@code dz_0 = 0}. At escape we report
 * {@code DE = |z| · log|z| / |dz|} — the standard Mandelbrot distance
 * estimator. In-set / cardioid / period-2 points report
 * {@link Double#NaN} for DE.
 *
 * <p>DE is opt-in because the derivative track adds 4 multiplies and 2 adds
 * per iteration. The kernel is rebuilt per render (cheap), so callers pay this
 * cost only when a colorizer or downstream consumer asked for the field.
 */
public final class MandelbrotKernel implements IterationKernel {

    private static final double LN2 = Math.log(2.0);

    private final int maxIter;
    private final double escR2;
    private final boolean smoothing;
    private final boolean computeDe;

    public MandelbrotKernel(MandelbrotParams p, boolean computeDe) {
        this.maxIter = p.maxIter();
        this.escR2 = p.escapeRadius() * p.escapeRadius();
        this.smoothing = p.smoothing();
        this.computeDe = computeDe;
    }

    @Override
    public boolean supportsDistanceEstimate() {
        return true;
    }

    @Override
    public void sample(double cRe, double cIm, FieldsOut out) {
        if (insideMainCardioid(cRe, cIm) || insidePeriod2Bulb(cRe, cIm)) {
            out.escapeTime = -1.0;
            out.distanceEstimate = Double.NaN;
            return;
        }

        double zRe = 0.0, zIm = 0.0;
        double dzRe = 0.0, dzIm = 0.0;
        double zRe2 = 0.0, zIm2 = 0.0;

        int iter = 0;
        while (iter < maxIter) {
            zRe2 = zRe * zRe;
            zIm2 = zIm * zIm;
            if (zRe2 + zIm2 > escR2) break;

            if (computeDe) {
                final double newDzRe = 2.0 * (zRe * dzRe - zIm * dzIm) + 1.0;
                final double newDzIm = 2.0 * (zRe * dzIm + zIm * dzRe);
                dzRe = newDzRe;
                dzIm = newDzIm;
            }

            final double zReNext = zRe2 - zIm2 + cRe;
            zIm = 2.0 * zRe * zIm + cIm;
            zRe = zReNext;
            iter++;
        }

        if (iter >= maxIter) {
            out.escapeTime = -1.0;
            out.distanceEstimate = Double.NaN;
            return;
        }

        if (smoothing) {
            final double mag2 = zRe * zRe + zIm * zIm;
            final double logZn = 0.5 * Math.log(mag2);
            out.escapeTime = iter + 1.0 - Math.log(logZn / LN2) / LN2;
        } else {
            out.escapeTime = iter;
        }

        if (computeDe) {
            final double zMag = Math.sqrt(zRe * zRe + zIm * zIm);
            final double dzMag = Math.sqrt(dzRe * dzRe + dzIm * dzIm);
            out.distanceEstimate = (dzMag > 0.0 && zMag > 0.0)
                    ? zMag * Math.log(zMag) / dzMag
                    : Double.NaN;
        }
    }

    /** Closed-form main cardioid: {@code q(q + (cRe - 0.25)) < 0.25 cIm²} with
     * {@code q = (cRe - 0.25)² + cIm²}. */
    private static boolean insideMainCardioid(double cRe, double cIm) {
        final double dx = cRe - 0.25;
        final double q = dx * dx + cIm * cIm;
        return q * (q + dx) < 0.25 * cIm * cIm;
    }

    /** Period-2 bulb: open disc of radius {@code 1/4} centred at {@code (-1, 0)}. */
    private static boolean insidePeriod2Bulb(double cRe, double cIm) {
        final double dx = cRe + 1.0;
        return dx * dx + cIm * cIm < 0.0625;
    }
}
