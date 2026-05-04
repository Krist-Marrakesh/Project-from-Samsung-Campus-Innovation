package com.example.myapplication.ui.components

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush

/**
 * Loading-state placeholder that mimics what the rendered fractal will
 * occupy. Crucial for *perceptual* responsiveness: the user sees a tile of
 * the right size in the right place the moment they tap, not a layout that
 * jumps when the bytes finally arrive.
 *
 * Implementation: a slow horizontal sweep gradient between two surface
 * variants, using ``rememberInfiniteTransition`` so the animation drives
 * itself off Compose's frame clock without a coroutine scope.
 */
@Composable
fun ShimmerBox(modifier: Modifier = Modifier) {
    val transition = rememberInfiniteTransition(label = "shimmer")
    val progress by transition.animateFloat(
        initialValue = -1f,
        targetValue = 2f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1500, easing = LinearEasing),
            repeatMode = RepeatMode.Restart,
        ),
        label = "shimmer-progress",
    )

    val base = MaterialTheme.colorScheme.surfaceContainerHigh
    val highlight = MaterialTheme.colorScheme.surfaceContainerHighest
    val brush = Brush.linearGradient(
        colors = listOf(base, highlight, base),
        // Sweep diagonal: each frame the gradient slides further along the
        // box. ``progress`` overshoots [0, 1] so the highlight fully exits
        // the visible area at each end of the cycle.
        start = Offset(progress * 1000f, 0f),
        end = Offset(progress * 1000f + 600f, 600f),
    )
    Box(modifier = modifier.background(brush))
}
