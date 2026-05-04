package com.example.myapplication.ui.screen

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.createSavedStateHandle
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.myapplication.FractalovApp
import com.example.myapplication.domain.FractalRecipe
import com.example.myapplication.domain.Presets
import com.example.myapplication.network.FractalovApi
import com.example.myapplication.ui.Routes
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.ClassDiscriminatorMode
import kotlinx.serialization.json.Json

/**
 * Variations grid state.
 *
 * Two phases:
 *   * ``recipes`` — the result of ``/ml/variations``. ML side returns
 *     recipes only; this phase resolves quickly (~one round-trip).
 *   * ``tiles`` — one entry per variation, each tracking its own
 *     loading/done/error state independently. We fan out parallel
 *     ``/render`` calls; tiles fade in as bytes arrive, regardless of
 *     order. Slow tiles don't block fast ones.
 */
data class VariationsUiState(
    val baseRecipe: FractalRecipe? = null,
    val isLoadingRecipes: Boolean = false,
    val recipesError: String? = null,
    val tiles: List<VariationTile> = emptyList(),
    val seed: Int = 0,
    val selectedTileIndex: Int? = null,
)

data class VariationTile(
    val recipe: FractalRecipe,
    val state: TileState,
)

sealed interface TileState {
    data object Loading : TileState
    data class Done(val bytes: ByteArray, val renderMs: Long) : TileState
    data class Error(val message: String) : TileState
}

class VariationsViewModel(
    savedStateHandle: SavedStateHandle,
    private val api: FractalovApi,
) : ViewModel() {

    private val _state = MutableStateFlow(VariationsUiState())
    val state: StateFlow<VariationsUiState> = _state.asStateFlow()

    init {
        val seedRecipeJson: String? = savedStateHandle[Routes.VARIATIONS_ARG]
        val seed = parseRecipe(seedRecipeJson) ?: Presets.ALL.first().recipe()
        _state.update { it.copy(baseRecipe = seed) }
        loadVariations(count = 8, rngSeed = 0)
    }

    /** Re-roll: fetch a fresh batch of variations under a new seed. */
    fun reroll() {
        val newSeed = _state.value.seed + 1
        loadVariations(count = 8, rngSeed = newSeed)
    }

    fun selectTile(index: Int?) {
        _state.update { it.copy(selectedTileIndex = index) }
    }

    private fun loadVariations(count: Int, rngSeed: Int) {
        val base = _state.value.baseRecipe ?: return
        _state.update {
            it.copy(
                isLoadingRecipes = true,
                recipesError = null,
                tiles = emptyList(),
                seed = rngSeed,
                selectedTileIndex = null,
            )
        }
        viewModelScope.launch {
            val recipes = runCatching { api.variations(base, count, rngSeed) }
            recipes.fold(
                onSuccess = { rs ->
                    val tiles = rs.map { VariationTile(it, TileState.Loading) }
                    _state.update { it.copy(isLoadingRecipes = false, tiles = tiles) }
                    renderTilesInParallel(rs)
                },
                onFailure = { e ->
                    _state.update {
                        it.copy(
                            isLoadingRecipes = false,
                            recipesError = e.message ?: e.javaClass.simpleName,
                        )
                    }
                },
            )
        }
    }

    /** Fan-out: every tile renders independently, in parallel. We update
     *  the tile's state as soon as its render returns — fast tiles appear
     *  first regardless of order in the recipe list. */
    private fun renderTilesInParallel(recipes: List<FractalRecipe>) {
        viewModelScope.launch {
            recipes.mapIndexed { index, recipe ->
                async {
                    val outcome = runCatching { api.render(recipe) }
                    _state.update { st ->
                        val updated = st.tiles.toMutableList()
                        if (index in updated.indices) {
                            updated[index] = updated[index].copy(
                                state = outcome.fold(
                                    onSuccess = { TileState.Done(it.pngBytes, it.renderMs) },
                                    onFailure = { TileState.Error(it.message ?: it.javaClass.simpleName) },
                                ),
                            )
                        }
                        st.copy(tiles = updated)
                    }
                }
            }.awaitAll()
        }
    }

    @OptIn(ExperimentalSerializationApi::class)
    private val recipeJson = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
        classDiscriminatorMode = ClassDiscriminatorMode.POLYMORPHIC
    }

    private fun parseRecipe(s: String?): FractalRecipe? =
        if (s.isNullOrBlank()) null
        else runCatching { recipeJson.decodeFromString(FractalRecipe.serializer(), s) }
            .getOrNull()

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app = (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as FractalovApp)
                VariationsViewModel(
                    savedStateHandle = createSavedStateHandle(),
                    api = app.api,
                )
            }
        }
    }
}
