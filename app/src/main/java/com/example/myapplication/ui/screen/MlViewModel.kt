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

data class MlUiState(
    val isLoading: Boolean = false,
    val payload: MlPayload = MlPayload.Idle,
)

sealed interface MlPayload {
    data object Idle : MlPayload
    data object Loading : MlPayload
    data class Done(
        val family: String,
        val confidence: Double,
        val cRe: Double?,
        val cIm: Double?,
        val renderId: String,
        val originalBytes: ByteArray,
        val bytes: ByteArray,
        val totalMs: Long,
        val recipe: FractalRecipe,
    ) : MlPayload
    data class Error(val message: String) : MlPayload
}

class MlViewModel(private val api: FractalovApi) : ViewModel() {

    private val _state = MutableStateFlow(MlUiState())
    val state: StateFlow<MlUiState> = _state.asStateFlow()

    fun runMlRender(bytes: ByteArray, fileName: String) {
        _state.update { it.copy(isLoading = true, payload = MlPayload.Loading) }
        viewModelScope.launch {
            val payload = runCatching { api.mlRenderFromImage(bytes, fileName) }
                .fold(
                    onSuccess = { r ->
                        MlPayload.Done(
                            family = r.family,
                            confidence = r.familyConfidence,
                            cRe = r.cRe,
                            cIm = r.cIm,
                            renderId = r.renderId,
                            originalBytes = r.originalBytes,
                            bytes = r.pngBytes,
                            totalMs = r.totalMs,
                            recipe = r.recipe,
                        )
                    },
                    onFailure = { MlPayload.Error(it.message ?: it.javaClass.simpleName) },
                )
            _state.update { it.copy(isLoading = false, payload = payload) }
        }
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app = (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as FractalovApp)
                MlViewModel(app.api)
            }
        }
    }
}
