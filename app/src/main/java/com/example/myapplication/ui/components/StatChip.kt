package com.example.myapplication.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.myapplication.ui.theme.MonoNumericSmall

/**
 * Compact label/value chip for read-only metadata.
 *
 * Shape is a rounded rectangle, not a Material chip — chips imply
 * interactivity, which is wrong here. The visual weight is on the value;
 * the label is a small caps-y prefix in the dim foreground.
 *
 * Use cases: render perf timings ("render 28 ms"), recipe knobs
 * ("iter 200", "ssaa 1"), confidence percentages.
 */
@Composable
fun StatChip(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    accent: Color? = null,
) {
    val outline = accent?.copy(alpha = 0.55f) ?: MaterialTheme.colorScheme.outlineVariant
    val background = accent?.copy(alpha = 0.10f)
        ?: MaterialTheme.colorScheme.surfaceContainerHighest
    Row(
        modifier = modifier
            .background(background, RoundedCornerShape(8.dp))
            .border(width = 1.dp, color = outline, shape = RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            style = MonoNumericSmall.copy(color = accent ?: MaterialTheme.colorScheme.onSurface),
        )
    }
}
