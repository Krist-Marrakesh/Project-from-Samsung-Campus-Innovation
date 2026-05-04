package com.example.myapplication.network

import com.example.myapplication.domain.FractalRecipe
import kotlinx.serialization.Serializable

// Wire types for the backend render + ML endpoints. Kept deliberately minimal:
// only the fields the Android client actually consumes. Unknown fields are
// tolerated by the Json configuration in FractalovApi; that lets the backend
// grow new metadata without forcing a client release.

@Serializable
data class RenderRequestDto(val recipe: FractalRecipe)

@Serializable
data class PerformanceBreakdownDto(
    val renderMs: Long,
    val colorizeMs: Long,
    val encodeMs: Long,
    val totalMs: Long,
)

@Serializable
data class RenderResponseDto(
    val requestId: String,
    val status: String,
    val imageBase64: String,
    val format: String,
    val widthPx: Int,
    val heightPx: Int,
    val performance: PerformanceBreakdownDto,
)

@Serializable
data class MlSuggestionDto(
    val family: String,
    val familyConfidence: Double,
    val cRe: Double? = null,
    val cIm: Double? = null,
    val suggestedRecipe: FractalRecipe,
)

@Serializable
data class RenderRecordDto(
    val id: String,
    val recipeId: String,
    val imageUrl: String,
    val widthPx: Int,
    val heightPx: Int,
    val paletteName: String,
    val colorMode: String,
    val samplesPerAxis: Int,
    val fileSizeBytes: Long,
    val performance: PerformanceBreakdownDto,
)

@Serializable
data class MlRenderResponseDto(
    val suggestion: MlSuggestionDto,
    val projectId: String,
    val recipeId: String,
    val render: RenderRecordDto,
)

@Serializable
data class MlVariationsRequestDto(
    val recipe: FractalRecipe,
    val count: Int,
    val seed: Int,
)

@Serializable
data class MlVariationsResponseDto(
    val recipes: List<FractalRecipe>,
)
