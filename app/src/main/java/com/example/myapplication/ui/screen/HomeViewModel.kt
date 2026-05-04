package com.example.myapplication.ui.screen

import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.myapplication.FractalovApp
import com.example.myapplication.domain.FractalRecipe
import com.example.myapplication.domain.Presets
import com.example.myapplication.domain.Viewport
import com.example.myapplication.network.FractalovApi
import com.example.myapplication.render.LocalFractalRenderer
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
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

    /** ``bitmap`` is the most recent render. ``isLowRes`` flags the
     *  fast-tier render that was issued during the gesture; once the
     *  high-res render lands the same payload is replaced with
     *  ``isLowRes = false``.
     *
     *  ``recipe`` is the *displayed* recipe — what the user thinks
     *  they're looking at. ``renderedViewport`` is what the bitmap
     *  actually contains (a 1.5× overrender so dragging shows
     *  pre-computed margin pixels rather than empty canvas). */
    data class Done(
        val bitmap: Bitmap,
        val widthPx: Int,
        val heightPx: Int,
        val renderMs: Long,
        val isLowRes: Boolean,
        val recipe: FractalRecipe,
        val renderedViewport: Viewport,
    ) : RenderPayload

    data class Error(val message: String) : RenderPayload
}

/**
 * Two-tier on-device renderer.
 *
 * The Java backend is the source of truth for the project, but during
 * pinch-and-pan we render locally so every gesture frame produces fresh
 * pixels rather than a stretched copy of a stale PNG. ``api`` is kept
 * available for screens that genuinely need the backend (Compare,
 * Variations, ML).
 *
 * Resolution policy:
 *   * **LIVE** — 512×512 with reduced ``maxIter`` (smoothing off),
 *     fired on every viewport change. ~15–40 ms on a flagship.
 *     Resolution chosen to match a typical preview canvas size (no
 *     visible up-scaling pixelation), iterations capped because the
 *     user can't see boundary detail mid-gesture anyway.
 *   * **HIGH** — 1024×1024 with the recipe's full ``maxIter`` and
 *     smoothing on, fired after the gesture has paused for
 *     ``HIGH_RES_DEBOUNCE_MS``. ~80–150 ms. Replaces the live bitmap.
 *
 * Cancellation: each new viewport supersedes the in-flight job.
 */
class HomeViewModel(
    @Suppress("unused") private val api: FractalovApi,
) : ViewModel() {

    private val _state = MutableStateFlow(HomeUiState())
    val state: StateFlow<HomeUiState> = _state.asStateFlow()

    private var liveJob: Job? = null
    private var highResJob: Job? = null

    fun selectFamily(key: String) {
        _state.update { it.copy(selectedFamilyKey = key) }
    }

    fun renderPreset() {
        val preset = Presets.ALL.firstOrNull { it.key == _state.value.selectedFamilyKey } ?: return
        kickOff(preset.recipe(), keepStaleBitmap = false)
    }

    /** Replace the current viewport with ``vp`` and re-render via the
     *  local renderer. The previous bitmap stays visible while the user
     *  is still gesturing, so the screen always has something on it. */
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
        liveJob?.cancel()
        highResJob?.cancel()
        if (!keepStaleBitmap) {
            _state.update { it.copy(isLoading = true, payload = RenderPayload.Loading) }
        } else {
            _state.update { it.copy(isLoading = true) }
        }

        // Both tiers render at the SAME resolution and the SAME
        // overrendered viewport (1.5× the visible area). Only iteration
        // count varies between tiers. This avoids both the "visible
        // resolution swap" bug AND the "grey edge during drag" bug.
        liveJob = viewModelScope.launch {
            val payload = renderInto(
                renderRecipe = recipe.withLiveTierDetail().withOverrender(OVERRENDER_FACTOR),
                displayRecipe = recipe,
                isLowRes = true,
            )
            _state.update { st ->
                if (payload is RenderPayload.Error) {
                    st.copy(isLoading = false, payload = payload)
                } else {
                    st.copy(payload = payload)
                }
            }

            // HIGH tier — same overrendered viewport, full maxIter.
            // Visible only as a sharpening of the existing image. If
            // the user keeps moving, the next kickOff cancels this job.
            highResJob = viewModelScope.launch {
                delay(HIGH_RES_DEBOUNCE_MS)
                val hi = renderInto(
                    renderRecipe = recipe.withOverrender(OVERRENDER_FACTOR),
                    displayRecipe = recipe,
                    isLowRes = false,
                )
                _state.update { it.copy(isLoading = false, payload = hi) }
            }
        }
    }

    private suspend fun renderInto(
        renderRecipe: FractalRecipe,
        displayRecipe: FractalRecipe,
        isLowRes: Boolean,
    ): RenderPayload {
        return runCatching {
            var bitmap: Bitmap? = null
            val ms = measureTimeMillis {
                bitmap = LocalFractalRenderer.render(renderRecipe, RENDER_RES, RENDER_RES)
            }
            RenderPayload.Done(
                bitmap = bitmap!!,
                widthPx = RENDER_RES,
                heightPx = RENDER_RES,
                renderMs = ms,
                isLowRes = isLowRes,
                recipe = displayRecipe,
                renderedViewport = renderRecipe.viewport,
            )
        }.getOrElse {
            RenderPayload.Error(it.message ?: it.javaClass.simpleName)
        }
    }

    companion object {
        // Single render resolution for both tiers — see kickOff() for why.
        // 768² on a flagship: live ≈ 50ms, high ≈ 120ms.
        private const val RENDER_RES = 768
        private const val HIGH_RES_DEBOUNCE_MS = 180L

        // Overrender margin: how much wider/taller the rendered viewport
        // is vs the visible canvas. 1.5× means each side has 25% of
        // the canvas worth of pre-computed pixels behind the bezel —
        // a finger pan up to 25% of canvas-width never exposes empty
        // space, while the renderer still only does 2.25× the pixel
        // work of a non-overrendered frame.
        private const val OVERRENDER_FACTOR: Double = 1.5

        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app = (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as FractalovApp)
                HomeViewModel(app.api)
            }
        }
    }
}

/**
 * Build the live-tier variant of a recipe.
 *
 * ``maxIter`` is reduced to 75% of the high-tier value (clamped to a
 * minimum of 80 — below that we start losing the spiral structure on
 * dense Julia/Multibrot recipes and the live tier looks like a "black
 * blob" rather than a recognisable fractal).
 *
 * Smoothing stays on. The ``log`` per escape pixel is one MUL + LUT
 * lookup in JIT-compiled Kotlin; the visual difference between
 * smoothing on and off is large (banded vs continuous gradient) and
 * the eye notices the change immediately when the high tier replaces
 * the live tier — that visible "redraw" is exactly the bug we're
 * trying to avoid.
 */
private fun com.example.myapplication.domain.FractalRecipe.withLiveTierDetail():
        com.example.myapplication.domain.FractalRecipe {
    fun reduce(maxIter: Int): Int = maxOf(80, (maxIter * 3) / 4)
    val newParams = when (val p = params) {
        is com.example.myapplication.domain.FractalParams.Mandelbrot ->
            p.copy(maxIter = reduce(p.maxIter))
        is com.example.myapplication.domain.FractalParams.Julia ->
            p.copy(maxIter = reduce(p.maxIter))
        is com.example.myapplication.domain.FractalParams.BurningShip ->
            p.copy(maxIter = reduce(p.maxIter))
        is com.example.myapplication.domain.FractalParams.Multibrot ->
            p.copy(maxIter = reduce(p.maxIter))
    }
    return copy(params = newParams)
}

/**
 * Expand the recipe's viewport by ``factor`` in each dimension while
 * keeping the centre fixed. The renderer produces a bitmap that
 * contains the visible viewport plus a ``(factor − 1) / 2`` margin
 * around it, which the gesture handler then exposes as needed without
 * triggering a fresh render.
 */
private fun com.example.myapplication.domain.FractalRecipe.withOverrender(
    factor: Double,
): com.example.myapplication.domain.FractalRecipe {
    val vp = viewport
    val cx = (vp.xMin + vp.xMax) / 2.0
    val cy = (vp.yMin + vp.yMax) / 2.0
    val halfW = (vp.xMax - vp.xMin) / 2.0 * factor
    val halfH = (vp.yMax - vp.yMin) / 2.0 * factor
    return copy(
        viewport = com.example.myapplication.domain.Viewport(
            xMin = cx - halfW,
            xMax = cx + halfW,
            yMin = cy - halfH,
            yMax = cy + halfH,
        ),
    )
}
