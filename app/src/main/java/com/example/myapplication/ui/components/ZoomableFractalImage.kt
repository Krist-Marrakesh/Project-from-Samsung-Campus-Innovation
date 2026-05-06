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

/**
 * Pinch-zoomable, drag-pannable fractal preview.
 *
 * Strategy: **optical only during gesture, single render on release**.
 *
 * While at least one finger is on the canvas, every motion updates the
 * local ``scale`` and ``offset`` state which drive a
 * :func:`Modifier.graphicsLayer` transform. No kernel work runs — the
 * user manipulates the existing bitmap directly. With
 * :class:`FilterQuality.Low` this looks softly blurred during zoom, not
 * pixelated, and there's nothing for the system to "snap" between.
 *
 * On the **release** of the last finger we compute the new analytic
 * viewport from the accumulated transform and call ``onCommitViewport``.
 * The host renders one fresh bitmap at full quality; when it lands the
 * local transform resets to identity and the user sees a sharp
 * high-resolution image.
 *
 * No intermediate renders, no live-tier vs high-tier swap, no jitter.
 *
 * @param baseViewport the viewport the *current* bitmap was rendered
 *        with. All gesture math is relative to this.
 * @param onCommitViewport invoked with the new analytic viewport when
 *        the gesture releases; the host should render a fresh bitmap.
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
    minScale: Float = 0.25f,
    maxScale: Float = 32f,
) {
    val shape = RoundedCornerShape(cornerRadius)
    var scale by remember { mutableStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    var canvasSize by remember { mutableStateOf(IntSize.Zero) }

    // When a fresh bitmap arrives (rendered for the viewport we asked
    // for), reset the local transform — gestures from now on are
    // relative to the new image.
    LaunchedEffect(bitmap) {
        scale = 1f
        offset = Offset.Zero
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
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = contentDescription,
                contentScale = ContentScale.Crop,
                // Bilinear filter so optical-only transformation during
                // a gesture looks soft rather than pixel-mosaic.
                filterQuality = FilterQuality.Low,
                modifier = Modifier
                    .matchParentSize()
                    .graphicsLayer {
                        scaleX = scale
                        scaleY = scale
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
                                    // Anchor compensation — the pixel
                                    // under the centroid stays put.
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
                                }
                            } while (event.changes.any { it.pressed })
                            // All fingers up → commit. No debounce here:
                            // the gesture loop only exits when the last
                            // pointer lifts, which is the right moment.
                            if (canvasSize != IntSize.Zero && baseViewport != null) {
                                val committed = computeViewport(
                                    base = baseViewport,
                                    scale = scale,
                                    offset = offset,
                                    canvasSize = canvasSize,
                                )
                                onCommitViewport(committed)
                            }
                        }
                    },
            )
        }
    }
}

/**
 * Convert local (scale, offset) into a fresh analytic viewport.
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
