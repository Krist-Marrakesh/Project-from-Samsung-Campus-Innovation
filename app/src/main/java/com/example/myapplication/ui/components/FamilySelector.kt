package com.example.myapplication.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.myapplication.domain.Presets

/**
 * Horizontal selector for the four families.
 *
 * Why a flex Row instead of a LazyRow:
 *   * The set of families is fixed and small (4); we want every family
 *     visible at all times rather than hidden behind a scroll affordance
 *     the user might not realise exists. Earlier the LazyRow clipped the
 *     fourth family on phone-width screens.
 *   * Equal-weight pills make the row feel like a segmented selector,
 *     which matches the "pick one of N modes" semantics. Real screen
 *     research has shown segmented controls outperform horizontal
 *     scrollers for tiny fixed-cardinality choices.
 *
 * Each pill carries a coloured leading dot keyed to the family's accent —
 * the same colour the user will see on every RecipeBadge and Variations
 * tile, so they build visual memory for "orange = Mandelbrot" without us
 * having to re-state it on every screen.
 */
@Composable
fun FamilySelector(
    currentKey: String,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Presets.ALL.forEach { preset ->
            FamilyPill(
                key = preset.key,
                title = preset.displayName,
                selected = preset.key == currentKey,
                onClick = { onSelect(preset.key) },
                modifier = Modifier.weight(1f),
            )
        }
    }
}

@Composable
private fun FamilyPill(
    key: String,
    title: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val accent = familyAccent(key)
    val targetBg = if (selected) accent.copy(alpha = 0.18f) else MaterialTheme.colorScheme.surfaceContainer
    val targetBorder = if (selected) accent else MaterialTheme.colorScheme.outlineVariant
    val targetText = if (selected) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant

    val bg by animateColorAsState(targetBg, label = "$key-bg")
    val border by animateColorAsState(targetBorder, label = "$key-border")
    val textColor by animateColorAsState(targetText, label = "$key-text")
    val borderWidth by animateDpAsState(if (selected) 1.5.dp else 1.dp, label = "$key-border-w")

    Row(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(bg, RoundedCornerShape(14.dp))
            .border(borderWidth, border, RoundedCornerShape(14.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 8.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp, Alignment.CenterHorizontally),
    ) {
        Dot(color = accent)
        Text(
            text = title,
            style = MaterialTheme.typography.labelLarge,
            color = textColor,
            // Long titles ("Burning Ship", "Multibrot N=5") fit at 4-up
            // on a phone width via labelLarge + ellipsis, with the dot
            // staying as the colour cue when the text gets truncated on
            // very narrow devices.
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
private fun Dot(color: Color) {
    Row(modifier = Modifier.size(8.dp).background(color, CircleShape)) {}
}
