package com.example.myapplication.ui.screen

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.GridView
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.outlined.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.myapplication.ui.components.ErrorState
import com.example.myapplication.ui.components.FractalImage
import com.example.myapplication.ui.components.PrimarySurface
import com.example.myapplication.ui.components.RecipeBadge
import com.example.myapplication.ui.components.StatChip
import com.example.myapplication.ui.theme.MonoNumeric
import kotlin.math.min

/**
 * Predicted-vs-reconstructed comparison.
 *
 * Layout adapts to orientation:
 *   * Portrait → vertical stack: original card on top, reconstructed below,
 *     with the prediction details band sandwiched between them. The eye
 *     reads top-down, so the model's commitment ("this is a Julia at
 *     c = …") appears next to the image it was made about.
 *   * Landscape / wide → side-by-side; the prediction band moves below the
 *     row so the comparison is the dominant visual.
 *
 * Loading model: original image appears as soon as the picker resolves;
 * the reconstructed slot shows a shimmer until the backend round-trip
 * completes, then crossfades to the rendered PNG. This is the perceptual
 * win — the user gets immediate feedback on their action even though the
 * full pipeline takes a second.
 */
@Composable
fun ComparisonScreen(
    onOpenVariations: (recipeJson: String) -> Unit,
    vm: ComparisonViewModel = viewModel(factory = ComparisonViewModel.Factory),
) {
    val state by vm.state.collectAsStateWithLifecycle()
    val context = LocalContext.current

    val pickImage = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
    ) { uri ->
        if (uri != null) {
            val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            if (bytes != null) {
                vm.submit(bytes, fileName = "input.png")
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = "Pick a fractal image. We render the model's predicted recipe and put the two side by side — the closer they match, the better the inverse-inference pass succeeded.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = {
                    pickImage.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                    )
                },
                enabled = !state.isLoading,
                modifier = Modifier.weight(1f),
            ) {
                if (state.isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                    Spacer(Modifier.size(8.dp))
                    Text("Reconstructing…")
                } else {
                    Icon(
                        imageVector = Icons.Outlined.PhotoLibrary,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(Modifier.size(8.dp))
                    Text(if (state.original == null) "Pick image" else "Pick another")
                }
            }
            if (state.original != null && !state.isLoading) {
                OutlinedButton(onClick = vm::reset) {
                    Icon(
                        imageVector = Icons.Outlined.Refresh,
                        contentDescription = "Reset",
                        modifier = Modifier.size(18.dp),
                    )
                }
            }
        }

        if (state.isLoading) {
            // Subtle progress bar at the top of the cards so the user has a
            // continuous signal of "still working". The CircularProgressIndicator
            // inside the button covers tap feedback; this covers wait feedback.
            LinearProgressIndicator(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(2.dp)
                    .clip(RoundedCornerShape(2.dp)),
                color = MaterialTheme.colorScheme.primary,
                trackColor = MaterialTheme.colorScheme.surfaceContainerHigh,
            )
        }

        when (val payload = state.payload) {
            is ComparisonPayload.Idle -> IdleHint()
            is ComparisonPayload.Loading -> LoadingPair(originalBytes = state.original)
            is ComparisonPayload.Done -> ResultPair(
                originalBytes = state.original,
                payload = payload,
                onOpenVariations = onOpenVariations,
            )
            is ComparisonPayload.Error -> {
                ErrorState(
                    title = "Reconstruction failed",
                    detail = payload.message,
                    onRetry = null,
                )
            }
        }

        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun IdleHint() {
    PrimarySurface {
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text(
                text = "No image picked yet.",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text = "Pick a fractal image — any of the dataset PNGs work, or a screenshot of a render.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
        }
    }
}

@Composable
private fun LoadingPair(originalBytes: ByteArray?) {
    PairLayout(
        original = {
            FractalImage(
                bytes = originalBytes,
                contentDescription = "user-supplied original",
                modifier = Modifier.fillMaxWidth(),
            )
        },
        details = {
            PrimarySurface(modifier = Modifier.fillMaxWidth(), contentPadding = 12.dp) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        text = "ML inference in progress",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Text(
                        text = "Predicting family + recipe parameters and re-rendering the suggestion on the Java backend.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        },
        reconstructed = {
            // null bytes ⇒ shimmer in the same slot the result will land in.
            FractalImage(
                bytes = null,
                contentDescription = "ML reconstruction (loading)",
                modifier = Modifier.fillMaxWidth(),
            )
        },
    )
}

@Composable
private fun ResultPair(
    originalBytes: ByteArray?,
    payload: ComparisonPayload.Done,
    onOpenVariations: (String) -> Unit,
) {
    PairLayout(
        original = {
            FractalImage(
                bytes = originalBytes,
                contentDescription = "user-supplied original",
                modifier = Modifier.fillMaxWidth(),
            )
        },
        details = {
            PredictionCard(
                payload = payload,
                onOpenVariations = onOpenVariations,
            )
        },
        reconstructed = {
            FractalImage(
                bytes = payload.reconstructed,
                contentDescription = "ML-reconstructed render",
                modifier = Modifier.fillMaxWidth(),
            )
        },
    )
}

@Composable
private fun PredictionCard(
    payload: ComparisonPayload.Done,
    onOpenVariations: (String) -> Unit,
) {
    PrimarySurface(modifier = Modifier.fillMaxWidth(), contentPadding = 14.dp) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            RecipeBadge(
                recipe = payload.recipe,
                confidence = payload.confidence,
            )

            ConfidenceBar(confidence = payload.confidence)

            // Mono-formatted regression numbers — the bit a researcher
            // actually wants to see.
            if (payload.cRe != null && payload.cIm != null) {
                Text(
                    text = "c = %.4f %+.4fi".format(payload.cRe, payload.cIm),
                    style = MonoNumeric.copy(color = MaterialTheme.colorScheme.onSurface),
                )
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                StatChip(label = "total", value = "${payload.totalMs} ms")
            }

            OutlinedButton(
                onClick = { onOpenVariations(serialiseRecipe(payload.recipe)) },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Icon(
                    imageVector = Icons.Outlined.GridView,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp),
                )
                Spacer(Modifier.size(8.dp))
                Text("Explore variations of this recipe")
            }
        }
    }
}

@Composable
private fun ConfidenceBar(confidence: Double) {
    val pct = (confidence.coerceIn(0.0, 1.0) * 100).toFloat()
    val color = when {
        confidence >= 0.85 -> MaterialTheme.colorScheme.primary
        confidence >= 0.6 -> MaterialTheme.colorScheme.tertiary
        else -> MaterialTheme.colorScheme.secondary
    }
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                text = "family confidence",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = "%.1f%%".format(pct),
                style = MonoNumeric.copy(color = color),
            )
        }
        LinearProgressIndicator(
            progress = { (pct / 100f).coerceIn(0f, 1f) },
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp)
                .clip(RoundedCornerShape(3.dp)),
            color = color,
            trackColor = MaterialTheme.colorScheme.surfaceContainerHighest,
        )
    }
}

/** Renders the three slots responsively. Wide screens (≥ 720 dp) put the
 *  image cards side by side; phones stack. The details card always sits
 *  in the middle (above the recon on phones, below the row on tablets). */
@Composable
private fun PairLayout(
    original: @Composable () -> Unit,
    details: @Composable () -> Unit,
    reconstructed: @Composable () -> Unit,
) {
    val configuration = LocalConfiguration.current
    val isWide = min(configuration.screenWidthDp, configuration.screenHeightDp) >= 600
    if (isWide) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Box(modifier = Modifier.weight(1f)) { original() }
                Box(modifier = Modifier.weight(1f)) { reconstructed() }
            }
            details()
        }
    } else {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            ImageHeader(text = "Original")
            original()
            ImageHeader(text = "Predicted by the model →")
            details()
            reconstructed()
        }
    }
}

@Composable
private fun ImageHeader(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}
