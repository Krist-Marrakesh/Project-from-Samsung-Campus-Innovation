package com.example.myapplication.ui.screen

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.clickable
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.myapplication.domain.FractalParams
import com.example.myapplication.domain.FractalRecipe
import com.example.myapplication.ui.components.ErrorState
import com.example.myapplication.ui.components.FractalImage
import com.example.myapplication.ui.components.PrimarySurface
import com.example.myapplication.ui.components.RecipeBadge
import com.example.myapplication.ui.components.StatChip
import com.example.myapplication.ui.theme.MonoNumeric

/**
 * Grid of recipes perturbed around a base. Each tile is independently
 * loading; tiles fade in as their render arrives, so the user sees
 * progress immediately rather than waiting for the whole batch.
 *
 * Tap a tile → bottom-sheet-style dialog with the full recipe inspector.
 *
 * @param seedRecipeJson serialised FractalRecipe passed as a navigation
 *        argument. Falls back to the first preset on null/parse failure
 *        — the screen is functional even when navigated to directly.
 */
@Composable
fun VariationsScreen(
    seedRecipeJson: String?,
    vm: VariationsViewModel = viewModel(factory = VariationsViewModel.Factory),
) {
    val state by vm.state.collectAsStateWithLifecycle()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Header(
            base = state.baseRecipe,
            isLoading = state.isLoadingRecipes,
            onReroll = vm::reroll,
        )

        when {
            state.recipesError != null -> {
                ErrorState(
                    title = "ML variations failed",
                    detail = state.recipesError,
                    onRetry = vm::reroll,
                )
            }
            state.tiles.isEmpty() && state.isLoadingRecipes -> {
                LoadingPlaceholder()
            }
            else -> {
                VariationGrid(
                    tiles = state.tiles,
                    onTap = { idx -> vm.selectTile(idx) },
                )
            }
        }

        Spacer(Modifier.height(8.dp))
    }

    val selectedIndex = state.selectedTileIndex
    if (selectedIndex != null && selectedIndex in state.tiles.indices) {
        val tile = state.tiles[selectedIndex]
        TileInspectorDialog(
            tile = tile,
            onDismiss = { vm.selectTile(null) },
        )
    }
}

@Composable
private fun Header(
    base: FractalRecipe?,
    isLoading: Boolean,
    onReroll: () -> Unit,
) {
    PrimarySurface(modifier = Modifier.fillMaxWidth(), contentPadding = 14.dp) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Eight perturbations of the base recipe",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Text(
                        text = "Tap any tile for the full recipe. Re-roll for a fresh batch.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                OutlinedButton(onClick = onReroll, enabled = !isLoading) {
                    if (isLoading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Outlined.AutoAwesome,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp),
                        )
                    }
                    Spacer(Modifier.size(6.dp))
                    Text("Re-roll")
                }
            }
            if (base != null) {
                RecipeBadge(recipe = base)
            }
        }
    }
}

@Composable
private fun LoadingPlaceholder() {
    PrimarySurface(modifier = Modifier.fillMaxWidth()) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            CircularProgressIndicator(
                modifier = Modifier.size(16.dp),
                strokeWidth = 2.dp,
                color = MaterialTheme.colorScheme.primary,
            )
            Text(
                text = "Asking the ML service for variations…",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun VariationGrid(
    tiles: List<VariationTile>,
    onTap: (Int) -> Unit,
) {
    val configuration = LocalConfiguration.current
    val cols = when {
        configuration.screenWidthDp >= 720 -> 4
        configuration.screenWidthDp >= 480 -> 3
        else -> 2
    }
    LazyVerticalGrid(
        columns = GridCells.Fixed(cols),
        contentPadding = PaddingValues(top = 4.dp, bottom = 24.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        userScrollEnabled = true,
    ) {
        items(items = tiles, key = { System.identityHashCode(it.recipe) }) { tile ->
            VariationTileView(
                tile = tile,
                onTap = { onTap(tiles.indexOf(tile)) },
            )
        }
    }
}

@Composable
private fun VariationTileView(
    tile: VariationTile,
    onTap: () -> Unit,
) {
    Box(modifier = Modifier.clip(RoundedCornerShape(16.dp))) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clickable { onTap() },
        ) {
            FractalImage(
                bytes = (tile.state as? TileState.Done)?.bytes,
                contentDescription = "variation tile",
                modifier = Modifier.fillMaxWidth(),
            )
            // Floating badge in the corner — reads off-image without
            // covering the focal area.
            AnimatedVisibility(
                visible = tile.state is TileState.Done,
                enter = fadeIn(),
                exit = fadeOut(),
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(6.dp),
            ) {
                StatChip(
                    label = "render",
                    value = "${(tile.state as? TileState.Done)?.renderMs ?: 0} ms",
                )
            }
            // Error overlay — shown when render of this individual tile failed.
            if (tile.state is TileState.Error) {
                Box(
                    modifier = Modifier
                        .align(Alignment.Center)
                        .padding(8.dp),
                ) {
                    Text(
                        text = "render failed",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
        }
    }
}

@Composable
private fun TileInspectorDialog(
    tile: VariationTile,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Close") }
        },
        title = {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Variation details", style = MaterialTheme.typography.titleLarge)
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Outlined.Close, contentDescription = "Close")
                }
            }
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                FractalImage(
                    bytes = (tile.state as? TileState.Done)?.bytes,
                    contentDescription = "selected variation",
                    modifier = Modifier.fillMaxWidth(),
                )
                RecipeInspector(recipe = tile.recipe)
            }
        },
        containerColor = MaterialTheme.colorScheme.surface,
    )
}

@Composable
private fun RecipeInspector(recipe: FractalRecipe) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        RecipeBadge(recipe = recipe)
        val vp = recipe.viewport
        Text(
            text = "viewport [%.3f .. %.3f] × [%.3f .. %.3f]"
                .format(vp.xMin, vp.xMax, vp.yMin, vp.yMax),
            style = MonoNumeric.copy(color = MaterialTheme.colorScheme.onSurface),
        )
        when (val p = recipe.params) {
            is FractalParams.Julia -> Text(
                text = "params  c = %.4f%+.4fi  ·  iter %d".format(p.cRe, p.cIm, p.maxIter),
                style = MonoNumeric.copy(color = MaterialTheme.colorScheme.onSurface),
            )
            is FractalParams.Multibrot -> Text(
                text = "params  N = ${p.exponent}  ·  iter ${p.maxIter}",
                style = MonoNumeric.copy(color = MaterialTheme.colorScheme.onSurface),
            )
            is FractalParams.Mandelbrot -> Text(
                text = "params  iter ${p.maxIter}  ·  esc² ${p.escapeRadius}",
                style = MonoNumeric.copy(color = MaterialTheme.colorScheme.onSurface),
            )
            is FractalParams.BurningShip -> Text(
                text = "params  iter ${p.maxIter}  ·  esc² ${p.escapeRadius}",
                style = MonoNumeric.copy(color = MaterialTheme.colorScheme.onSurface),
            )
        }
        Text(
            text = "palette ${recipe.colorSettings.paletteName}  ·  mode ${recipe.colorSettings.mode}",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
