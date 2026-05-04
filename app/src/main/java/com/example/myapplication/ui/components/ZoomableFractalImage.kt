package com.example.myapplication.ui.components

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.example.myapplication.domain.Viewport
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Pinch-zoomable, drag-pannable fractal preview with **overrender margin**.
 *
 * The bitmap supplied in ``bitmap`` was rendered at a viewport
 * ``overrenderFactor`` times wider/taller than what the user is
 * supposed to see. The Image is sized to that overrendered area
 * (centred over the canvas Box, clipped by the Box's bounds), so:
 *
 *   * In rest, the user sees only the central portion of the bitmap
 *     — exactly the recipe's logical viewport.
 *   * Dragging in any direction up to ``(overrenderFactor - 1) / 2``
 *     of canvas size pulls in pixels that were already rendered into
 *     the margin. There is no "grey edge of the world".
 *   * Pinch + drag still trigger a re-render via ``onCommitViewport``
 *     after a brief debounce; the new bitmap fully replaces the old
 *     one with a fresh overrender margin around the new viewport.
 *
 * @param baseViewport the *visible* viewport (NOT the overrendered
 *        one). Gesture math is relative to this. Pass the recipe's
 *        own viewport here.
 * @param overrenderFactor must match the host's overrender — see
 *        :class:`HomeViewModel.OVERRENDER_FACTOR`.
 */
@Composable
fun ZoomableFractalImage(
    bitmap: Bitmap?,
    baseViewport: Viewport?,
    contentDescription: String,
    onCommitViewport: (Viewport) -> Unit,
    modifier: Modifier = Modifier,
    aspectRatio: Float = 1f,
    cornerRadius: Dp = 16.dp,
    debounceMs: Long = 50,
    minScale: Float = 0.25f,
    maxScale: Float = 32f,
    overrenderFactor: Float = 1.5f,
) {
    val shape = RoundedCornerShape(cornerRadius)
    var scale by remember { mutableStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    var canvasSize by remember { mutableStateOf(IntSize.Zero) }

    LaunchedEffect(bitmap) {
        scale = 1f
        offset = Offset.Zero
    }

    val scope = rememberCoroutineScope()
    var debounceJob: Job? by remember { mutableStateOf<Job?>(null) }

    fun scheduleCommit() {
        if (bitmap == null || baseViewport == null || canvasSize == IntSize.Zero) return
        debounceJob?.cancel()
        debounceJob = scope.launch {
            delay(debounceMs)
            val committed = computeViewport(baseViewport, scale, offset, canvasSize)
            onCommitViewport(committed)
        }
    }

    Box(
        modifier = modifier
            .aspectRatio(aspectRatio)
            .clip(shape)
            .background(MaterialTheme.colorScheme.surfaceContainerHigh)
            .border(
                width = 1.dp,
                color = MaterialTheme.colorScheme.outlineVariant,
                shape = shape,
            )
            .onSizeChanged { canvasSize = it },
        contentAlignment = Alignment.Center,
    ) {
        if (bitmap == null) {
            ShimmerBox(modifier = Modifier.matchParentSize())
        } else {
            // Inner image is sized to the OVERRENDERED area. Box clips
            // anything that extends past canvas bounds. Without a
            // gesture, only the central 1/overrenderFactor² is visible —
            // exactly the recipe's visible viewport. Drag uncovers
            // already-rendered margin.
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = contentDescription,
                contentScale = ContentScale.Fit,
                filterQuality = FilterQuality.Low,
                modifier = Modifier
                    .matchParentSize()
                    .graphicsLayer {
                        // Initial scale = overrenderFactor so the
                        // visible part of the rendered area maps to
                        // canvas size. User scale stacks on top.
                        scaleX = overrenderFactor * scale
                        scaleY = overrenderFactor * scale
                        translationX = offset.x
                        translationY = offset.y
                    }
                    .pointerInput(baseViewport) {
                        awaitEachGesture {
                            awaitFirstDown(requireUnconsumed = false)
                            do {
                                val event = awaitPointerEvent(PointerEventPass.Main)
                                val zoomChange = event.calculateZoom()
                                val panChange = event.calculatePan()
                                if (zoomChange != 1f || panChange != Offset.Zero) {
                                    val newScale = (scale * zoomChange)
                                        .coerceIn(minScale, maxScale)
                                    val centroid = event.calculateCentroid(useCurrent = true)
                                    val w = canvasSize.width.toFloat()
                                    val h = canvasSize.height.toFloat()
                                    val zoomDelta = newScale / scale.coerceAtLeast(1e-6f)
                                    val centerX = w / 2f
                                    val centerY = h / 2f
                                    val anchorDx = (centroid.x - centerX - offset.x) * (1f - zoomDelta)
                                    val anchorDy = (centroid.y - centerY - offset.y) * (1f - zoomDelta)
                                    scale = newScale
                                    offset = Offset(
                                        x = offset.x + panChange.x + anchorDx,
                                        y = offset.y + panChange.y + anchorDy,
                                    )
                                    if (event.changes.any { it.positionChanged() }) {
                                        event.changes.forEach { it.consume() }
                                    }
                                    scheduleCommit()
                                }
                            } while (event.changes.any { it.pressed })
                            scheduleCommit()
                        }
                    },
            )
        }
    }
}

/**
 * Convert local (scale, offset) into a fresh analytic viewport.
 *
 * ``base`` here is the *visible* viewport — what the user sees in the
 * canvas right now. The bitmap may extend beyond that into a margin,
 * but the gesture math operates in canvas space, so the visible
 * viewport is the right reference.
 *
 *     screen_centre_in_viewport = base_centre - offset / canvas_w * base_span / scale
 *     new_span                  = base_span / scale
 *
 * y-axis is flipped because the rendered image runs y_max at row 0.
 */
internal fun computeViewport(
    base: Viewport,
    scale: Float,
    offset: Offset,
    canvasSize: IntSize,
): Viewport {
    val s = scale.coerceAtLeast(1e-6f)
    val w = canvasSize.width.coerceAtLeast(1).toFloat()
    val h = canvasSize.height.coerceAtLeast(1).toFloat()

    val baseSpanX = base.xMax - base.xMin
    val baseSpanY = base.yMax - base.yMin
    val newSpanX = baseSpanX / s
    val newSpanY = baseSpanY / s

    val baseCx = (base.xMin + base.xMax) / 2.0
    val baseCy = (base.yMin + base.yMax) / 2.0
    val newCx = baseCx - (offset.x / w) * baseSpanX / s
    val newCy = baseCy + (offset.y / h) * baseSpanY / s

    return Viewport(
        xMin = newCx - newSpanX / 2.0,
        xMax = newCx + newSpanX / 2.0,
        yMin = newCy - newSpanY / 2.0,
        yMax = newCy + newSpanY / 2.0,
    )
}
