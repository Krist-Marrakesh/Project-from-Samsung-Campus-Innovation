package com.example.myapplication.ui.screen

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.material.icons.outlined.GridView
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.myapplication.ui.components.ErrorState
import com.example.myapplication.ui.components.FractalImage
import com.example.myapplication.ui.components.PrimarySurface
import com.example.myapplication.ui.components.RecipeBadge
import com.example.myapplication.ui.components.StatChip

@Composable
fun MlScreen(
    onOpenVariations: (recipeJson: String) -> Unit,
    vm: MlViewModel = viewModel(factory = MlViewModel.Factory),
) {
    val state by vm.state.collectAsStateWithLifecycle()
    val context = LocalContext.current

    val pickImage = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
    ) { uri ->
        if (uri != null) {
            val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            if (bytes != null) {
                vm.runMlRender(bytes, fileName = "input.png")
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
            text = "Upload a fractal image. The Java backend forwards it to the Python ML service, which classifies the family and predicts the recipe parameters; the backend then renders the predicted recipe and persists the result.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Button(
            onClick = {
                pickImage.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                )
            },
            enabled = !state.isLoading,
            modifier = Modifier.fillMaxWidth(),
        ) {
            if (state.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp,
                    color = MaterialTheme.colorScheme.onPrimary,
                )
                Spacer(Modifier.size(8.dp))
                Text("Inferring + rendering…")
            } else {
                Icon(
                    imageVector = Icons.Outlined.PhotoLibrary,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp),
                )
                Spacer(Modifier.size(8.dp))
                Text("Pick image")
            }
        }

        when (val payload = state.payload) {
            is MlPayload.Idle -> {
                PrimarySurface {
                    Text(
                        text = "Pick an image from your gallery (any fractal-ish render works) and the model will guess the recipe behind it.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            is MlPayload.Loading -> {
                PrimarySurface(contentPadding = 12.dp) {
                    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        FractalImage(
                            bytes = null,
                            contentDescription = "ML inference in progress",
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Text(
                            text = "Uploading → inference → render → fetch persisted PNG.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
            is MlPayload.Done -> {
                PrimarySurface(contentPadding = 12.dp) {
                    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        RecipeBadge(
                            recipe = payload.recipe,
                            confidence = payload.confidence,
                        )
                        FractalImage(
                            bytes = payload.bytes,
                            contentDescription = "ML-reconstructed render",
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            StatChip(
                                label = "render id",
                                value = payload.renderId.take(8),
                            )
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
            is MlPayload.Error -> {
                ErrorState(
                    title = "Inference failed",
                    detail = payload.message,
                    onRetry = null,
                )
            }
        }

        Spacer(Modifier.height(24.dp))
    }
}
