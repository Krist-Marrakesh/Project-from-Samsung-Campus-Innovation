package com.example.myapplication.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

/**
 * App-wide theme entry point.
 *
 * Default-dark: fractals look better on the midnight palette and the dataset
 * imagery this app stares at all day is high-contrast, high-saturation. We
 * still respect a light-mode override; users with platform light mode get
 * a Material 3 light scheme that is intentionally less polished — the
 * design budget is on dark.
 *
 * Dynamic colour: opt-in on API 31+. Some users have heavily tinted dynamic
 * palettes that wash out the purple/cyan accents; ``preferDynamicColor=false``
 * (default) sticks to our hand-tuned palette so the app looks consistent
 * across devices. Flip to ``true`` per-screen for an experiment.
 */
@Composable
fun FractalovTheme(
    darkTheme: Boolean = isSystemInDarkTheme() || true,   // default dark, see above
    preferDynamicColor: Boolean = false,
    content: @Composable () -> Unit,
) {
    val colors = when {
        preferDynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val ctx = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = Color.Transparent.toArgb()
            window.navigationBarColor = Color.Transparent.toArgb()
            val controller = WindowCompat.getInsetsController(window, view)
            controller.isAppearanceLightStatusBars = !darkTheme
            controller.isAppearanceLightNavigationBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colors,
        typography = FractalovTypography,
        content = content,
    )
}

private val DarkColors = darkColorScheme(
    primary = PrimaryAccent,
    onPrimary = OnPrimaryAccent,
    primaryContainer = PrimaryAccentMuted,
    onPrimaryContainer = OnPrimaryAccent,

    secondary = SecondaryAccent,
    onSecondary = OnSecondaryAccent,
    secondaryContainer = SecondaryAccentMuted,
    onSecondaryContainer = OnSecondaryAccent,

    tertiary = TertiaryAccent,
    onTertiary = OnTertiaryAccent,

    background = MidnightBase,
    onBackground = Foreground,

    surface = MidnightSurface,
    onSurface = Foreground,
    surfaceContainer = MidnightSurfaceHi,
    surfaceContainerHigh = MidnightSurfaceVar,
    surfaceContainerHighest = MidnightSurfaceVar,
    surfaceVariant = MidnightSurfaceVar,
    onSurfaceVariant = ForegroundDim,
    outline = MidnightOutline,
    outlineVariant = MidnightOutline,

    error = ErrorAccent,
    onError = OnErrorAccent,
)

private val LightColors = lightColorScheme(
    primary = PrimaryAccentMuted,
    onPrimary = Color.White,
    secondary = SecondaryAccentMuted,
    onSecondary = Color.White,
    tertiary = TertiaryAccent,
    onTertiary = Color.White,
    error = ErrorAccent,
)
