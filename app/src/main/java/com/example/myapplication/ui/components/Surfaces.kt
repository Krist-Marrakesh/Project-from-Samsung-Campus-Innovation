package com.example.myapplication.ui.components

import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * The de facto card on this app.
 *
 * Material 3's stock ``Card`` is fine but the default elevation tonal
 * overlay reads as "lifted off a light page" — wrong cue on a dark theme.
 * ``PrimarySurface`` uses ``surfaceContainer`` (a flat midnight tint that
 * sits 1 step above the background) plus a 1 dp outline. The result feels
 * native to the dark palette and cleanly separates content blocks without
 * the "candy bar" look elevation gives on dark.
 */
@Composable
fun PrimarySurface(
    modifier: Modifier = Modifier,
    contentPadding: Dp = 16.dp,
    cornerRadius: Dp = 18.dp,
    content: @Composable () -> Unit,
) {
    val shape = RoundedCornerShape(cornerRadius)
    Surface(
        modifier = modifier
            .border(
                width = 1.dp,
                color = MaterialTheme.colorScheme.outlineVariant,
                shape = shape,
            ),
        shape = shape,
        color = MaterialTheme.colorScheme.surfaceContainer,
    ) {
        Box(modifier = Modifier.padding(contentPadding)) {
            content()
        }
    }
}
