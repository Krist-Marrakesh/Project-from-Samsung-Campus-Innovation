package com.example.myapplication.render

import android.graphics.Bitmap
import com.example.myapplication.domain.FractalParams
import com.example.myapplication.domain.FractalRecipe
import com.example.myapplication.domain.Viewport
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.ln

/**
 * On-device fractal renderer. The Java backend stays the source of truth
 * (and the only thing the ML pipelines touch), but for the home-screen
 * zoom/pan loop we can't afford a network round-trip per gesture frame —
 * and we definitely can't ``graphicsLayer``-stretch a 512×512 PNG to 8x
 * zoom without it pixelating into pure squares.
 *
 * This module produces fresh pixels on every viewport change. Two-tier
 * caller pattern (LIVE vs IDLE resolution) belongs to the ViewModel; the
 * renderer itself just takes ``widthPx × heightPx`` and produces a
 * ``Bitmap``.
 *
 * Performance plan (Snapdragon 8 Gen 2, 8 cores, 200 max_iter):
 *   * 256×256 Mandelbrot ≈ 8–15 ms — comfortable for live gestures
 *   * 1024×1024 Mandelbrot ≈ 80–150 ms — fine for after-gesture polish
 *
 * Parallelism is by row across ``Dispatchers.Default``'s thread pool.
 * Each pixel is independent so the only synchronisation is on the
 * shared ``IntArray`` (one write per pixel from a single owning row).
 */
object LocalFractalRenderer {

    /**
     * Render a recipe into an ARGB_8888 Bitmap of the requested size.
     *
     * @param recipe complete recipe (viewport + params + colour settings).
     *               Only the family-internal parameters and the viewport
     *               are honoured; ``colorSettings.mode`` is currently
     *               always treated as LINEAR. HISTOGRAM and DE modes
     *               require either a full sort or a derivative track —
     *               cheap on the JVM, less obviously cheap on a phone
     *               under finger-driven update pressure, so they stay
     *               on the backend for now.
     */
    suspend fun render(
        recipe: FractalRecipe,
        widthPx: Int,
        heightPx: Int,
    ): Bitmap = withContext(Dispatchers.Default) {
        require(widthPx > 0 && heightPx > 0) { "widthPx/heightPx must be positive" }
        val pixels = IntArray(widthPx * heightPx)
        val palette = Palette.byName(recipe.colorSettings.paletteName)
        val params = recipe.params
        val maxIter = paramsMaxIter(params)
        val esc2 = paramsEscapeRadius2(params)
        val smoothing = paramsSmoothing(params)
        val viewport = recipe.viewport

        // Block-sliced parallelism: ``cores`` workers each take every
        // ``cores``-th row. Avoids lock-step on a shared row counter and
        // gives a balanced workload because rows of similar y-coord have
        // similar iteration cost (the set is roughly horizontally
        // symmetric).
        val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(2)
        coroutineScope {
            (0 until cores).map { worker ->
                async {
                    var y = worker
                    while (y < heightPx) {
                        renderRow(
                            y = y,
                            widthPx = widthPx,
                            heightPx = heightPx,
                            viewport = viewport,
                            params = params,
                            maxIter = maxIter,
                            esc2 = esc2,
                            smoothing = smoothing,
                            palette = palette,
                            outPixels = pixels,
                        )
                        y += cores
                    }
                }
            }.awaitAll()
        }

        val bm = Bitmap.createBitmap(widthPx, heightPx, Bitmap.Config.ARGB_8888)
        bm.setPixels(pixels, /* offset= */ 0, /* stride= */ widthPx, 0, 0, widthPx, heightPx)
        bm
    }

    private fun renderRow(
        y: Int,
        widthPx: Int,
        heightPx: Int,
        viewport: Viewport,
        params: FractalParams,
        maxIter: Int,
        esc2: Double,
        smoothing: Boolean,
        palette: Palette,
        outPixels: IntArray,
    ) {
        // y-axis is yMax at row 0, yMin at the last row — same convention
        // as the Java GridSweep / backend renderer so a recipe rendered
        // on either side produces an image with the same orientation.
        val xMin = viewport.xMin
        val xSpan = viewport.xMax - viewport.xMin
        val yMax = viewport.yMax
        val ySpan = viewport.yMax - viewport.yMin
        val wDenom = if (widthPx > 1) (widthPx - 1).toDouble() else 1.0
        val hDenom = if (heightPx > 1) (heightPx - 1).toDouble() else 1.0
        val cIm = yMax - ySpan * y / hDenom
        val rowOffset = y * widthPx
        for (x in 0 until widthPx) {
            val cRe = xMin + xSpan * x / wDenom
            val escapeTime = when (params) {
                is FractalParams.Mandelbrot -> mandelbrot(cRe, cIm, maxIter, esc2, smoothing)
                is FractalParams.Julia -> julia(
                    cRe, cIm,
                    cValRe = params.cRe,
                    cValIm = params.cIm,
                    maxIter = maxIter,
                    esc2 = esc2,
                    smoothing = smoothing,
                )
                is FractalParams.BurningShip -> burningShip(cRe, cIm, maxIter, esc2, smoothing)
                is FractalParams.Multibrot -> multibrot(
                    cRe, cIm,
                    n = params.exponent,
                    maxIter = maxIter,
                    esc2 = esc2,
                    smoothing = smoothing,
                )
            }
            outPixels[rowOffset + x] = colorize(escapeTime, maxIter, palette)
        }
    }

    // ----------------------- iteration kernels ---------------------------

    /** Returns -1.0 for in-set, otherwise smooth/integer escape value. */
    private fun mandelbrot(cRe: Double, cIm: Double, maxIter: Int, esc2: Double, smoothing: Boolean): Double {
        // Cardioid + period-2 bulb early bail. Same closed-form tests as
        // the Java MandelbrotKernel; cheap upfront cost saves the entire
        // iteration loop for ~50% of points on whole-set views.
        val dxC = cRe - 0.25
        val q = dxC * dxC + cIm * cIm
        if (q * (q + dxC) < 0.25 * cIm * cIm) return -1.0
        val dxB = cRe + 1.0
        if (dxB * dxB + cIm * cIm < 0.0625) return -1.0

        var zRe = 0.0
        var zIm = 0.0
        var zRe2 = 0.0
        var zIm2 = 0.0
        var iter = 0
        while (iter < maxIter) {
            zRe2 = zRe * zRe
            zIm2 = zIm * zIm
            if (zRe2 + zIm2 > esc2) break
            val zReNext = zRe2 - zIm2 + cRe
            zIm = 2.0 * zRe * zIm + cIm
            zRe = zReNext
            iter++
        }
        if (iter >= maxIter) return -1.0
        if (!smoothing) return iter.toDouble()
        val mag2 = zRe * zRe + zIm * zIm
        return iter + 1.0 - ln(0.5 * ln(mag2) / LN2) / LN2
    }

    private fun julia(
        zRe0: Double, zIm0: Double,
        cValRe: Double, cValIm: Double,
        maxIter: Int, esc2: Double, smoothing: Boolean,
    ): Double {
        var zRe = zRe0
        var zIm = zIm0
        var zRe2 = zRe * zRe
        var zIm2 = zIm * zIm
        var iter = 0
        while (iter < maxIter) {
            if (zRe2 + zIm2 > esc2) break
            val zReNext = zRe2 - zIm2 + cValRe
            zIm = 2.0 * zRe * zIm + cValIm
            zRe = zReNext
            zRe2 = zRe * zRe
            zIm2 = zIm * zIm
            iter++
        }
        if (iter >= maxIter) return -1.0
        if (!smoothing) return iter.toDouble()
        return iter + 1.0 - ln(0.5 * ln(zRe2 + zIm2) / LN2) / LN2
    }

    private fun burningShip(cRe: Double, cIm: Double, maxIter: Int, esc2: Double, smoothing: Boolean): Double {
        var zRe = 0.0
        var zIm = 0.0
        var zRe2 = 0.0
        var zIm2 = 0.0
        var iter = 0
        while (iter < maxIter) {
            zRe2 = zRe * zRe
            zIm2 = zIm * zIm
            if (zRe2 + zIm2 > esc2) break
            val zReNext = zRe2 - zIm2 + cRe
            zIm = 2.0 * abs(zRe * zIm) + cIm
            zRe = zReNext
            iter++
        }
        if (iter >= maxIter) return -1.0
        if (!smoothing) return iter.toDouble()
        val mag2 = zRe * zRe + zIm * zIm
        return iter + 1.0 - ln(0.5 * ln(mag2) / LN2) / LN2
    }

    private fun multibrot(
        cRe: Double, cIm: Double,
        n: Int,
        maxIter: Int, esc2: Double, smoothing: Boolean,
    ): Double {
        // Integer N: compute z^N as (N - 1) complex multiplications via
        // a small specialised loop. Roughly 6× faster than the polar
        // form (atan2 + pow + cos + sin) per iteration on this hot path.
        // The dataset uses integer exponents in [2..10] so we never
        // need fractional N here.
        val logN = ln(n.toDouble())
        var zRe = 0.0
        var zIm = 0.0
        var iter = 0
        while (iter < maxIter) {
            val zMag2 = zRe * zRe + zIm * zIm
            if (zMag2 > esc2) break
            // z^N via repeated complex multiplication.
            //   start with (zRe, zIm)
            //   multiply (n-1) times by (zRe, zIm)
            // Saves one allocation pair vs binary exponentiation; for
            // the small N we need (≤10) the linear loop wins anyway.
            var pRe = zRe
            var pIm = zIm
            var k = 1
            while (k < n) {
                val newRe = pRe * zRe - pIm * zIm
                val newIm = pRe * zIm + pIm * zRe
                pRe = newRe
                pIm = newIm
                k++
            }
            zRe = pRe + cRe
            zIm = pIm + cIm
            iter++
        }
        val finalMag2 = zRe * zRe + zIm * zIm
        if (iter >= maxIter) return -1.0
        if (!smoothing) return iter.toDouble()
        return iter + 1.0 - ln(0.5 * ln(finalMag2) / logN) / logN
    }

    // ---------------------------- colorize -------------------------------

    private fun colorize(escapeTime: Double, maxIter: Int, palette: Palette): Int {
        if (escapeTime < 0.0) return IN_SET_ARGB
        val t = (escapeTime / maxIter).coerceIn(0.0, 1.0)
        return palette.argbAt(t)
    }

    // --------------------- params accessors (sealed) ---------------------

    private fun paramsMaxIter(p: FractalParams): Int = when (p) {
        is FractalParams.Mandelbrot -> p.maxIter
        is FractalParams.Julia -> p.maxIter
        is FractalParams.BurningShip -> p.maxIter
        is FractalParams.Multibrot -> p.maxIter
    }

    private fun paramsEscapeRadius2(p: FractalParams): Double = when (p) {
        is FractalParams.Mandelbrot -> p.escapeRadius * p.escapeRadius
        is FractalParams.Julia -> p.escapeRadius * p.escapeRadius
        is FractalParams.BurningShip -> p.escapeRadius * p.escapeRadius
        is FractalParams.Multibrot -> p.escapeRadius * p.escapeRadius
    }

    private fun paramsSmoothing(p: FractalParams): Boolean = when (p) {
        is FractalParams.Mandelbrot -> p.smoothing
        is FractalParams.Julia -> p.smoothing
        is FractalParams.BurningShip -> p.smoothing
        is FractalParams.Multibrot -> p.smoothing
    }

    private const val IN_SET_ARGB: Int = 0xFF000000.toInt()
    private val LN2 = ln(2.0)
}
