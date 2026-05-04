package com.example.myapplication.render

import kotlin.math.cos
import kotlin.math.roundToInt

/**
 * Pre-baked palette LUTs that mirror the four palettes registered in the
 * Java backend (``grayscale`` / ``fire`` / ``ocean`` / ``rainbow_cyclic``).
 *
 * We pre-compute a 256-entry ARGB integer table per palette at class init
 * time, so the per-pixel lookup is one multiply + one array read. The
 * control-point layout is hand-tuned to match the backend palettes
 * visually — no attempt at byte-identical equality (the backend uses
 * float-precision interp at the pixel level, we use the LUT). Acceptable
 * for the live-zoom path; if a researcher wants reference-quality colour
 * they're already going through the backend.
 */
class Palette private constructor(private val lut: IntArray) {

    fun argbAt(t: Double): Int {
        val idx = (t.coerceIn(0.0, 1.0) * (lut.size - 1)).roundToInt()
        return lut[idx]
    }

    companion object {
        private const val LUT_SIZE = 256

        fun byName(name: String): Palette = when (name) {
            "grayscale" -> Grayscale
            "fire" -> Fire
            "ocean" -> Ocean
            "rainbow_cyclic" -> RainbowCyclic
            else -> Fire   // graceful default rather than an exception
        }

        // Five-stop fire ramp: black → maroon → red → orange → yellow → white.
        private val Fire = build {
            stop(0.00, 0x00, 0x00, 0x00)
            stop(0.20, 0x40, 0x00, 0x00)
            stop(0.40, 0xC0, 0x10, 0x00)
            stop(0.60, 0xFF, 0x80, 0x00)
            stop(0.80, 0xFF, 0xE0, 0x00)
            stop(1.00, 0xFF, 0xFF, 0xFF)
        }

        // Ocean ramp: midnight blue → deep blue → cyan → light cyan → white.
        private val Ocean = build {
            stop(0.00, 0x00, 0x00, 0x10)
            stop(0.25, 0x00, 0x10, 0x60)
            stop(0.55, 0x00, 0x80, 0xC0)
            stop(0.80, 0x80, 0xE0, 0xFF)
            stop(1.00, 0xFF, 0xFF, 0xFF)
        }

        private val Grayscale = build {
            stop(0.0, 0x00, 0x00, 0x00)
            stop(1.0, 0xFF, 0xFF, 0xFF)
        }

        // Cyclic rainbow via three sinusoids 120° apart — the backend's
        // RainbowCyclicPalette uses the same construction. Cycles three
        // times across the [0, 1] LUT so escape gradients show banding
        // structure instead of a single hue-rotation.
        private val RainbowCyclic = run {
            val lut = IntArray(LUT_SIZE)
            val cycles = 3.0
            for (i in 0 until LUT_SIZE) {
                val t = i / (LUT_SIZE - 1.0) * cycles
                val phase = t * 2.0 * Math.PI
                val r = ((cos(phase) + 1.0) * 0.5 * 255.0).toInt().coerceIn(0, 255)
                val g = ((cos(phase + 2.0 * Math.PI / 3) + 1.0) * 0.5 * 255.0).toInt().coerceIn(0, 255)
                val b = ((cos(phase + 4.0 * Math.PI / 3) + 1.0) * 0.5 * 255.0).toInt().coerceIn(0, 255)
                lut[i] = packArgb(0xFF, r, g, b)
            }
            Palette(lut)
        }

        // ----- builder -----

        private fun build(block: Builder.() -> Unit): Palette {
            val builder = Builder()
            builder.block()
            return Palette(builder.materialise())
        }

        private class Builder {
            private data class Stop(val t: Double, val r: Int, val g: Int, val b: Int)
            private val stops = mutableListOf<Stop>()

            fun stop(t: Double, r: Int, g: Int, b: Int) {
                stops += Stop(t, r, g, b)
            }

            fun materialise(): IntArray {
                check(stops.size >= 2) { "palette needs at least two stops" }
                check(stops.first().t == 0.0 && stops.last().t == 1.0) {
                    "palette stops must span [0, 1]"
                }
                val sorted = stops.sortedBy { it.t }
                val lut = IntArray(LUT_SIZE)
                var segStart = 0
                for (i in 0 until LUT_SIZE) {
                    val t = i / (LUT_SIZE - 1.0)
                    while (segStart < sorted.size - 2 && t > sorted[segStart + 1].t) segStart++
                    val a = sorted[segStart]
                    val b = sorted[segStart + 1]
                    val span = (b.t - a.t).coerceAtLeast(1e-12)
                    val frac = ((t - a.t) / span).coerceIn(0.0, 1.0)
                    val r = (a.r + (b.r - a.r) * frac).toInt()
                    val g = (a.g + (b.g - a.g) * frac).toInt()
                    val bb = (a.b + (b.b - a.b) * frac).toInt()
                    lut[i] = packArgb(0xFF, r, g, bb)
                }
                return lut
            }
        }

        private fun packArgb(a: Int, r: Int, g: Int, b: Int): Int =
            (a and 0xFF shl 24) or (r and 0xFF shl 16) or (g and 0xFF shl 8) or (b and 0xFF)
    }
}
