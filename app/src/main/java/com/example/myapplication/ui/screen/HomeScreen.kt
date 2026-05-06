package com.example.myapplication.ui.screen

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.CenterFocusStrong
import androidx.compose.material.icons.outlined.GridView
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.myapplication.domain.FractalRecipe
import com.example.myapplication.domain.Presets
import com.example.myapplication.ui.components.ErrorState
import com.example.myapplication.ui.components.FamilySelector
import com.example.myapplication.ui.components.FractalImage
import com.example.myapplication.ui.components.PrimarySurface
import com.example.myapplication.ui.components.StatChip
import com.example.myapplication.ui.components.ZoomableFractalImage
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.ClassDiscriminatorMode
import kotlinx.serialization.json.Json

@Composable
fun HomeScreen(
    onOpenMl: () -> Unit,
    onOpenComparison: () -> Unit,
    onOpenVariations: (recipeJson: String) -> Unit,
    vm: HomeViewModel = viewModel(factory = HomeViewModel.Factory),
) {
    val state by vm.state.collectAsStateWithLifecycle()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Hero()

        FamilySelector(
            currentKey = state.selectedFamilyKey,
            onSelect = vm::selectFamily,
        )

        // The preview is the centrepiece. FractalImage holds the layout
        // even before bytes arrive, so the page never reflows on first
        // render. We use a 16:9 aspect ratio in idle state so there is
        // less wasted blank canvas, then expand to the natural 1:1 for
        // the actual fractal once the user has tapped Render.
        val previewAspect = if (state.payload is RenderPayload.Done) 1f else 16f / 9f
        PrimarySurface(contentPadding = 12.dp) {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                val done = state.payload as? RenderPayload.Done
                if (done != null) {
                    // Live, gesture-navigable preview. Pinch to zoom,
                    // drag to pan; on-device renderer regenerates fresh
                    // pixels via vm.navigateTo() so the image stays
                    // sharp at every zoom level — no PNG stretching.
                    ZoomableFractalImage(
                        bitmap = done.bitmap,
                        baseViewport = done.recipe.viewport,
                        contentDescription = "render of ${state.selectedFamilyKey}",
                        onCommitViewport = vm::navigateTo,
                        modifier = Modifier.fillMaxWidth(),
                        aspectRatio = previewAspect,
                    )
                    RenderStats(
                        widthPx = done.widthPx,
                        heightPx = done.heightPx,
                        renderMs = done.renderMs,
                    )
                    Text(
                        text = "Pinch to zoom · drag to pan · pixels re-computed on device.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    // Idle / loading / error → static placeholder
                    // (shimmer if loading) without gesture handlers.
                    FractalImage(
                        bitmap = null,
                        contentDescription = "preset render placeholder",
                        modifier = Modifier.fillMaxWidth(),
                        aspectRatio = previewAspect,
                    )
                    if (state.payload is RenderPayload.Idle) {
                        Text(
                            text = "Pick a family above and tap Render to call the backend's /render endpoint.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }

        ActionButtons(
            isLoading = state.isLoading,
            hasDoneRender = state.payload is RenderPayload.Done,
            currentRecipe = state.payload.recipeOrFallback(state.selectedFamilyKey),
            onRender = vm::renderPreset,
            onResetView = vm::resetViewport,
            onOpenMl = onOpenMl,
            onOpenComparison = onOpenComparison,
            onOpenVariations = onOpenVariations,
        )

        if (state.payload is RenderPayload.Error) {
            ErrorState(
                title = "Backend rejected the render",
                detail = (state.payload as RenderPayload.Error).message,
                onRetry = vm::renderPreset,
            )
        }

        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun Hero() {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
            text = "Generate, classify, reconstruct.",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Text(
            text = "Pick a fractal family, render it on the Java backend, then explore variations or upload a photo for ML inverse inference.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun RenderStats(widthPx: Int, heightPx: Int, renderMs: Long) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        StatChip(label = "size", value = "${widthPx}×${heightPx}")
        StatChip(label = "render", value = "$renderMs ms")
    }
}

@Composable
private fun ActionButtons(
    isLoading: Boolean,
    hasDoneRender: Boolean,
    currentRecipe: FractalRecipe,
    onRender: () -> Unit,
    onResetView: () -> Unit,
    onOpenMl: () -> Unit,
    onOpenComparison: () -> Unit,
    onOpenVariations: (String) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        // Render is the primary CTA on first launch. Once we have an
        // image and the user has been zooming around, the more useful
        // primary action is "reset to the family's default viewport" —
        // a single icon-button to the right of Render.
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onRender,
                enabled = !isLoading,
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = MaterialTheme.colorScheme.onPrimary,
                ),
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                    Spacer(Modifier.size(8.dp))
                    Text("Rendering…")
                } else {
                    Text(if (hasDoneRender) "Re-render" else "Render")
                }
            }
            if (hasDoneRender) {
                OutlinedButton(
                    onClick = onResetView,
                    enabled = !isLoading,
                ) {
                    Icon(
                        imageVector = Icons.Outlined.CenterFocusStrong,
                        contentDescription = "Reset viewport",
                        modifier = Modifier.size(18.dp),
                    )
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(
                onClick = onOpenComparison,
                modifier = Modifier.weight(1f),
            ) {
                Icon(
                    imageVector = Icons.Outlined.Image,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp),
                )
                Spacer(Modifier.size(8.dp))
                Text("Compare")
            }
            OutlinedButton(
                onClick = { onOpenVariations(serialiseRecipe(currentRecipe)) },
                modifier = Modifier.weight(1f),
            ) {
                Icon(
                    imageVector = Icons.Outlined.GridView,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp),
                )
                Spacer(Modifier.size(8.dp))
                Text("Variations")
            }
        }

        OutlinedButton(
            onClick = onOpenMl,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Icon(
                imageVector = Icons.Outlined.AutoAwesome,
                contentDescription = null,
                modifier = Modifier.size(18.dp),
            )
            Spacer(Modifier.size(8.dp))
            Text("ML from image →")
        }
    }
}

/** Pick the recipe to feed to Variations. After a successful render we use
 *  the actual rendered recipe; otherwise fall back to the selected family's
 *  preset so the button remains usable on first launch. */
private fun RenderPayload.recipeOrFallback(selectedFamilyKey: String): FractalRecipe {
    val rendered = (this as? RenderPayload.Done)?.recipe
    if (rendered != null) return rendered
    val preset = Presets.ALL.firstOrNull { it.key == selectedFamilyKey } ?: Presets.ALL.first()
    return preset.recipe()
}

@OptIn(ExperimentalSerializationApi::class)
private val recipeJson = Json {
    encodeDefaults = true
    classDiscriminatorMode = ClassDiscriminatorMode.POLYMORPHIC
}

internal fun serialiseRecipe(recipe: FractalRecipe): String =
    recipeJson.encodeToString(FractalRecipe.serializer(), recipe)
