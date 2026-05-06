package com.example.myapplication.ui.screen

import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.myapplication.FractalovApp
import com.example.myapplication.domain.FractalParams
import com.example.myapplication.domain.FractalRecipe
import com.example.myapplication.domain.Presets
import com.example.myapplication.domain.Viewport
import com.example.myapplication.network.FractalovApi
import com.example.myapplication.render.LocalFractalRenderer
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.system.measureTimeMillis

data class HomeUiState(
    val selectedFamilyKey: String = "mandelbrot",
    val isLoading: Boolean = false,
    val payload: RenderPayload = RenderPayload.Idle,
)

sealed interface RenderPayload {
    data object Idle : RenderPayload
    data object Loading : RenderPayload
    data class Done(
        val bitmap: Bitmap,
        val widthPx: Int,
        val heightPx: Int,
        val renderMs: Long,
        val recipe: FractalRecipe,
    ) : RenderPayload
    data class Error(val message: String) : RenderPayload
}

/**
 * Single-render-on-release renderer.
 *
 * Earlier iterations tried a two-tier (live + high) strategy where the
 * live tier re-rendered every gesture frame. That produced two complaints:
 *   * "дёргается" — every ~50 ms a new bitmap arrived, the local
 *     ``graphicsLayer`` transform reset to identity, and the user saw
 *     a discontinuous "snap" between successive renders.
 *   * "пикселит" — Multi/Burning kernels couldn't keep up at 768², the
 *     live render fell to 512², and the bitmap was stretched onto a
 *     ~1080×1080 canvas at integer scale, exposing the rendering grid.
 *
 * The corrected strategy:
 *   * **During gesture**: zero kernel work. The user's pinch / drag is
 *     applied via :func:`Modifier.graphicsLayer` on the existing high-
 *     resolution bitmap. There are no intermediate renders, so there is
 *     nothing to "snap" or jitter.
 *   * **On gesture release** (after a short debounce so micro-pauses
 *     mid-pan don't trigger spurious renders): one full-quality render
 *     at ``RENDER_RES`` (1024² by default), with the recipe's full
 *     ``maxIter`` and smoothing on. ``RENDER_RES`` is twice the
 *     typical canvas size in pixels, so the bitmap is retina-density —
 *     no visible aliasing during steady viewing.
 *
 * The trade-off: while the user is zooming aggressively, the visible
 * image is the *previous* render stretched/transformed by
 * ``graphicsLayer``. With ``FilterQuality.Low`` (bilinear) this looks
 * blurred, not pixelated. ~100 ms after lift the new full-resolution
 * render takes over.
 */
class HomeViewModel(
    @Suppress("unused") private val api: FractalovApi,
) : ViewModel() {

    private val _state = MutableStateFlow(HomeUiState())
    val state: StateFlow<HomeUiState> = _state.asStateFlow()

    private var renderJob: Job? = null

    fun selectFamily(key: String) {
        _state.update { it.copy(selectedFamilyKey = key) }
    }

    fun renderPreset() {
        val preset = Presets.ALL.firstOrNull { it.key == _state.value.selectedFamilyKey } ?: return
        kickOff(preset.recipe(), keepStaleBitmap = false)
    }

    /** Replace the current viewport with ``vp`` and re-render. The
     *  caller (typically :class:`ZoomableFractalImage`) only invokes this
     *  on gesture release, so we don't need additional debouncing here. */
    fun navigateTo(vp: Viewport) {
        val current = currentRecipe() ?: return
        kickOff(current.copy(viewport = vp), keepStaleBitmap = true)
    }

    /** Reset to the family's default starting viewport. */
    fun resetViewport() {
        renderPreset()
    }

    private fun currentRecipe(): FractalRecipe? {
        return (_state.value.payload as? RenderPayload.Done)?.recipe
            ?: Presets.ALL.firstOrNull { it.key == _state.value.selectedFamilyKey }?.recipe?.invoke()
    }

    private fun kickOff(recipe: FractalRecipe, keepStaleBitmap: Boolean) {
        renderJob?.cancel()
        if (!keepStaleBitmap) {
            _state.update { it.copy(isLoading = true, payload = RenderPayload.Loading) }
        } else {
            // Keep the existing bitmap visible while the new one
            // computes — the gesture handler can still display the
            // optical-only transformed previous render.
            _state.update { it.copy(isLoading = true) }
        }
        val res = renderResFor(recipe)
        renderJob = viewModelScope.launch {
            val payload: RenderPayload = try {
                var bitmap: Bitmap? = null
                val ms = measureTimeMillis {
                    bitmap = LocalFractalRenderer.render(recipe, res, res)
                }
                // After the (suspending) render call, check that we
                // weren't superseded mid-flight. If we were, propagate
                // the cancellation rather than committing the stale
                // bitmap to the UI state.
                ensureActive()
                RenderPayload.Done(
                    bitmap = bitmap!!,
                    widthPx = res,
                    heightPx = res,
                    renderMs = ms,
                    recipe = recipe,
                )
            } catch (ce: CancellationException) {
                // Legitimate supersession: the user issued a fresher
                // request while this one was running. Don't surface as
                // an error — the new job will populate the UI shortly.
                throw ce
            } catch (t: Throwable) {
                RenderPayload.Error(t.message ?: t.javaClass.simpleName)
            }
            _state.update { it.copy(isLoading = false, payload = payload) }
        }
    }

    /** Per-family render resolution. The cheap kernels
     *  (Mandelbrot/Julia: one complex multiply per iteration) get the
     *  full retina-density 1536². Burning Ship's ``abs()`` step makes
     *  it ~30% pricier, so we drop it to 1152². Multibrot N=5 is the
     *  heaviest — N−1 complex multiplies per iteration, ~4× the work
     *  of Mandelbrot — so it stays at 1024² to keep render time
     *  under ~250 ms even on mid-tier devices. */
    private fun renderResFor(recipe: FractalRecipe): Int = when (recipe.params) {
        is FractalParams.Multibrot -> 1024
        is FractalParams.BurningShip -> 1152
        else -> 1536
    }

    companion object {

        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app = (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as FractalovApp)
                HomeViewModel(app.api)
            }
        }
    }
}
