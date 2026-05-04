package com.example.myapplication.domain

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonClassDiscriminator

/**
 * Wire-shape mirror of the Java backend's FractalRecipe. The discriminator
 * is read from the recipe-level `fractalType` field — same as in the backend's
 * `@JsonTypeInfo(EXTERNAL_PROPERTY)` setup, but kotlinx.serialization expresses
 * the same idea by tagging the polymorphic field with a class discriminator
 * and serialising the enum into the parent object directly.
 *
 * The simplest cross-language scheme that survives both Jackson on the server
 * and kotlinx.serialization on the client is to keep `fractalType` at the
 * recipe level and key the params subtype off it. We mark the sealed type
 * with a kotlinx-side `@JsonClassDiscriminator` of "fractalType" so the
 * serialised payload says `"fractalType": "julia"` once, matching the
 * backend's wire format.
 */
@Serializable
data class FractalRecipe(
    val viewport: Viewport,
    val renderSettings: RenderSettings,
    val colorSettings: ColorSettings,
    val fractalType: String,
    val params: FractalParams,
)

@Serializable
data class Viewport(
    val xMin: Double,
    val xMax: Double,
    val yMin: Double,
    val yMax: Double,
)

@Serializable
data class RenderSettings(
    val widthPx: Int,
    val heightPx: Int,
    val samplesPerAxis: Int = 1,
)

@Serializable
data class ColorSettings(
    val paletteName: String,
    val mode: String = "linear",
)

@OptIn(ExperimentalSerializationApi::class)
@Serializable
@JsonClassDiscriminator("fractalType")
sealed interface FractalParams {

    @Serializable
    @SerialName("mandelbrot")
    data class Mandelbrot(
        val maxIter: Int,
        val escapeRadius: Double,
        val smoothing: Boolean,
    ) : FractalParams

    @Serializable
    @SerialName("julia")
    data class Julia(
        val cRe: Double,
        val cIm: Double,
        val maxIter: Int,
        val escapeRadius: Double,
        val smoothing: Boolean,
    ) : FractalParams

    @Serializable
    @SerialName("burning_ship")
    data class BurningShip(
        val maxIter: Int,
        val escapeRadius: Double,
        val smoothing: Boolean,
    ) : FractalParams

    @Serializable
    @SerialName("multibrot")
    data class Multibrot(
        val exponent: Int,
        val maxIter: Int,
        val escapeRadius: Double,
        val smoothing: Boolean,
    ) : FractalParams
}
