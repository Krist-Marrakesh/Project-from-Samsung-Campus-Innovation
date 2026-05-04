package com.example.myapplication.ui.components

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Static fractal preview surface.
 *
 * Two overloads:
 *   * ``FractalImage(bitmap: Bitmap?, …)`` — used by the on-device
 *     renderer path on the home screen. Bitmap is already in memory.
 *   * ``FractalImage(bytes: ByteArray?, …)`` — used by screens that
 *     get a PNG from the backend (Comparison, Variations, ML). The
 *     bytes are decoded once via ``BitmapFactory.decodeByteArray`` and
 *     remembered against the byte-array reference.
 *
 * Sizing: outer Box is bounded by ``modifier`` (typically
 * ``Modifier.fillMaxWidth()``) plus ``aspectRatio``. Children pin to
 * that bounded outer Box via ``Modifier.matchParentSize()``. Earlier
 * versions tried ``AnimatedVisibility`` / ``Crossfade`` here and hit
 * an infinite-measure loop on Compose BOM 2024.10 when nested inside a
 * scrolling parent — the simple if/else is the most predictable shape.
 */
@Composable
fun FractalImage(
    bitmap: Bitmap?,
    contentDescription: String,
    modifier: Modifier = Modifier,
    aspectRatio: Float = 1f,
    cornerRadius: Dp = 16.dp,
    filterQuality: FilterQuality = FilterQuality.Low,
) {
    val shape = RoundedCornerShape(cornerRadius)
    Box(
        modifier = modifier
            .aspectRatio(aspectRatio)
            .clip(shape)
            .background(MaterialTheme.colorScheme.surfaceContainerHigh)
            .border(
                width = 1.dp,
                color = MaterialTheme.colorScheme.outlineVariant,
                shape = shape,
            ),
        contentAlignment = Alignment.Center,
    ) {
        if (bitmap == null) {
            ShimmerBox(modifier = Modifier.matchParentSize())
        } else {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = contentDescription,
                contentScale = ContentScale.Crop,
                filterQuality = filterQuality,
                modifier = Modifier.matchParentSize(),
            )
        }
    }
}

/** PNG-bytes overload. Decodes once per byte-array reference. */
@Composable
fun FractalImage(
    bytes: ByteArray?,
    contentDescription: String,
    modifier: Modifier = Modifier,
    aspectRatio: Float = 1f,
    cornerRadius: Dp = 16.dp,
    filterQuality: FilterQuality = FilterQuality.Low,
) {
    val bitmap = remember(bytes) {
        bytes?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
    }
    FractalImage(
        bitmap = bitmap,
        contentDescription = contentDescription,
        modifier = modifier,
        aspectRatio = aspectRatio,
        cornerRadius = cornerRadius,
        filterQuality = filterQuality,
    )
}
