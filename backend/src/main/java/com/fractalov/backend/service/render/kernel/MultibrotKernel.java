package com.fractalov.backend.service.render.kernel;

import com.fractalov.backend.dto.MultibrotParams;
import com.fractalov.backend.service.render.FieldsOut;
import com.fractalov.backend.service.render.IterationKernel;

/**
 * Multibrot iteration kernel: {@code z_{n+1} = z_n^N + c} with {@code z_0 = 0},
 * integer {@code N ≥ 2}. {@code N = 2} recovers Mandelbrot.
 *
 * <p>{@code z^N} is computed in polar form ({@code atan2 + pow + cos/sin}) which
 * is several times more expensive per iteration than the Mandelbrot closed form
 * but covers any {@code N} with one branch. The smoothing formula generalises
 * with {@code log N} instead of {@code log 2}.
 *
 * <h3>Distance estimation</h3>
 * Recurrence: {@code dz_{n+1}/dc = N · z_n^{N-1} · dz_n + 1}, {@code dz_0 = 0}.
 * {@code z_n^{N-1}} is recovered cheaply from the same polar conversion that
 * computes {@code z_n^N}: {@code r^{N-1} = r^N / r}, {@code (N-1)θ = Nθ − θ}.
 * The {@code z = 0} edge case (only at iteration 0) is special-cased so the
 * division by {@code r} stays safe.
 */
public final class MultibrotKernel implements IterationKernel {

    private final int maxIter;
    private final double escR2;
    private final boolean smoothing;
    private final boolean computeDe;
    private final int exponent;
    private final double logExponent;

    public MultibrotKernel(MultibrotParams p, boolean computeDe) {
        this.maxIter = p.maxIter();
        this.escR2 = p.escapeRadius() * p.escapeRadius();
        this.smoothing = p.smoothing();
        this.computeDe = computeDe;
        this.exponent = p.exponent();
        this.logExponent = Math.log(exponent);
    }

    @Override
    public boolean supportsDistanceEstimate() {
        return true;
    }

    @Override
    public void sample(double cRe, double cIm, FieldsOut out) {
        double zRe = 0.0, zIm = 0.0;
        double dzRe = 0.0, dzIm = 0.0;
        double mag2 = 0.0;

        int iter = 0;
        while (iter < maxIter) {
            mag2 = zRe * zRe + zIm * zIm;
            if (mag2 > escR2) break;

            final double r = Math.sqrt(mag2);
            final double theta = Math.atan2(zIm, zRe);
            final double rN = Math.pow(r, exponent);
            final double nTheta = exponent * theta;
            final double zReNew = rN * Math.cos(nTheta) + cRe;
            final double zImNew = rN * Math.sin(nTheta) + cIm;

            if (computeDe) {
                if (r > 0.0) {
                    final double rNm1 = rN / r;
                    final double nm1Theta = nTheta - theta;
                    final double zPowNm1Re = rNm1 * Math.cos(nm1Theta);
                    final double zPowNm1Im = rNm1 * Math.sin(nm1Theta);
                    final double newDzRe = exponent * (zPowNm1Re * dzRe - zPowNm1Im * dzIm) + 1.0;
                    final double newDzIm = exponent * (zPowNm1Re * dzIm + zPowNm1Im * dzRe);
                    dzRe = newDzRe;
                    dzIm = newDzIm;
                } else {
                    // z = 0 only happens at iter 0 (z_0); N·z^{N-1}·dz vanishes for N > 1.
                    dzRe = 1.0;
                    dzIm = 0.0;
                }
            }

            zRe = zReNew;
            zIm = zImNew;
            iter++;
        }

        if (iter >= maxIter) {
            out.escapeTime = -1.0;
            out.distanceEstimate = Double.NaN;
            return;
        }

        if (smoothing) {
            final double logZn = 0.5 * Math.log(zRe * zRe + zIm * zIm);
            out.escapeTime = iter + 1.0 - Math.log(logZn / logExponent) / logExponent;
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
}
