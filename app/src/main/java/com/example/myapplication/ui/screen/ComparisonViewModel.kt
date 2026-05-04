package com.example.myapplication.ui.screen

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.myapplication.FractalovApp
import com.example.myapplication.domain.FractalRecipe
import com.example.myapplication.network.FractalovApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** Two-stage UI state: the moment the user picks an image we can show it
 *  immediately as ``original`` while the reconstruction is still in flight.
 *  This is the core perceptual-speed lever for this screen. */
data class ComparisonUiState(
    val original: ByteArray? = null,
    val isLoading: Boolean = false,
    val payload: ComparisonPayload = ComparisonPayload.Idle,
)

sealed interface ComparisonPayload {
    data object Idle : ComparisonPayload
    data object Loading : ComparisonPayload
    data class Done(
        val family: String,
        val confidence: Double,
        val cRe: Double?,
        val cIm: Double?,
        val reconstructed: ByteArray,
        val totalMs: Long,
        val recipe: FractalRecipe,
    ) : ComparisonPayload
    data class Error(val message: String) : ComparisonPayload
}

class ComparisonViewModel(private val api: FractalovApi) : ViewModel() {

    private val _state = MutableStateFlow(ComparisonUiState())
    val state: StateFlow<ComparisonUiState> = _state.asStateFlow()

    fun submit(originalBytes: ByteArray, fileName: String) {
        // Show the original immediately — the reconstructed slot will fill
        // when the network completes. No "loading" flash on the user's
        // chosen image.
        _state.update {
            it.copy(
                original = originalBytes,
                isLoading = true,
                payload = ComparisonPayload.Loading,
            )
        }
        viewModelScope.launch {
            val payload = runCatching { api.mlRenderFromImage(originalBytes, fileName) }
                .fold(
                    onSuccess = { r ->
                        ComparisonPayload.Done(
                            family = r.family,
                            confidence = r.familyConfidence,
                            cRe = r.cRe,
                            cIm = r.cIm,
                            reconstructed = r.pngBytes,
                            totalMs = r.totalMs,
                            recipe = r.recipe,
                        )
                    },
                    onFailure = { ComparisonPayload.Error(it.message ?: it.javaClass.simpleName) },
                )
            _state.update { it.copy(isLoading = false, payload = payload) }
        }
    }

    fun reset() {
        _state.value = ComparisonUiState()
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app = (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as FractalovApp)
                ComparisonViewModel(app.api)
            }
        }
    }
}
