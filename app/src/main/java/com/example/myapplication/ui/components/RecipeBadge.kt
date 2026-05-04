package com.example.myapplication.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.myapplication.domain.FractalParams
import com.example.myapplication.domain.FractalRecipe
import com.example.myapplication.ui.theme.MonoNumericSmall

/**
 * One-line summary of a recipe — family pill on the left, condensed key
 * params on the right. Drops into the corner of an image tile so a grid of
 * variations stays self-describing without over-decoration.
 */
@Composable
fun RecipeBadge(
    recipe: FractalRecipe,
    modifier: Modifier = Modifier,
    confidence: Double? = null,
) {
    Row(
        modifier = modifier
            .background(
                color = MaterialTheme.colorScheme.surfaceContainer.copy(alpha = 0.92f),
                shape = RoundedCornerShape(12.dp),
            )
            .border(
                width = 1.dp,
                color = MaterialTheme.colorScheme.outlineVariant,
                shape = RoundedCornerShape(12.dp),
            )
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        FamilyDot(recipe.fractalType)
        Column(verticalArrangement = Arrangement.spacedBy(1.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text(
                    text = displayName(recipe.fractalType),
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                if (confidence != null) {
                    Text(
                        text = "%.0f%%".format(confidence * 100),
                        style = MonoNumericSmall.copy(
                            color = MaterialTheme.colorScheme.tertiary,
                        ),
                    )
                }
            }
            keyParams(recipe)?.let { details ->
                Text(
                    text = details,
                    style = MonoNumericSmall.copy(
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    ),
                )
            }
        }
    }
}

/** Coloured bullet that visually keys each family. Order matches the
 *  alphabetic family encoder used by the model side. */
@Composable
private fun FamilyDot(family: String) {
    val color = familyAccent(family)
    Box(
        modifier = Modifier
            .size(8.dp)
            .background(color, CircleShape),
    )
}

@Composable
fun RecipeBadgeRow(
    recipe: FractalRecipe,
    modifier: Modifier = Modifier,
    confidence: Double? = null,
) {
    Row(modifier.fillMaxWidth()) {
        RecipeBadge(recipe, confidence = confidence)
    }
}

@Composable
internal fun familyAccent(family: String): Color = when (family) {
    "mandelbrot" -> MaterialTheme.colorScheme.secondary       // warm orange
    "julia" -> MaterialTheme.colorScheme.primary               // cyan
    "burning_ship" -> MaterialTheme.colorScheme.error          // muted red
    "multibrot" -> MaterialTheme.colorScheme.tertiary          // violet
    else -> MaterialTheme.colorScheme.onSurfaceVariant
}

internal fun displayName(family: String): String = when (family) {
    "mandelbrot" -> "Mandelbrot"
    "julia" -> "Julia"
    "burning_ship" -> "Burning Ship"
    "multibrot" -> "Multibrot"
    else -> family
}

/** Family-specific knobs in compact mono. */
private fun keyParams(recipe: FractalRecipe): String? = when (val p = recipe.params) {
    is FractalParams.Julia -> "c = %.3f%+.3fi".format(p.cRe, p.cIm)
    is FractalParams.Multibrot -> "N = ${p.exponent}  ·  iter ${p.maxIter}"
    is FractalParams.Mandelbrot -> "iter ${p.maxIter}"
    is FractalParams.BurningShip -> "iter ${p.maxIter}"
}
