package com.example.myapplication.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * Hand-tuned palette for the dark theme. Built around two design intents:
 *
 *   1. Fractals look best on a deep, slightly warm midnight background — pure
 *      black eats the smoothing gradients near the boundary, mid-grey washes
 *      out the interior. We pick a near-black with a very faint warm tint.
 *   2. The four built-in palettes (fire / ocean / rainbow / grayscale) span
 *      warm orange and cool cyan; the accent colours echo that range so a
 *      chip showing "fire" feels of-a-piece with the render itself.
 *
 * Light theme is fine to lift from Material's defaults — the app is rarely
 * used in light mode in practice (you're staring at dense imagery) and the
 * polish budget goes further on dark.
 */

// Surfaces
val MidnightBase = Color(0xFF0B0E13)
val MidnightSurface = Color(0xFF12161D)
val MidnightSurfaceHi = Color(0xFF1A2029)
val MidnightSurfaceVar = Color(0xFF252C36)
val MidnightOutline = Color(0xFF323A45)

// Text + on-surface
val Foreground = Color(0xFFE6E9EE)
val ForegroundDim = Color(0xFF98A0AD)

// Primary — saturated cool cyan, stands out on midnight.
val PrimaryAccent = Color(0xFF7DD3FC)
val PrimaryAccentMuted = Color(0xFF38BDF8)
val OnPrimaryAccent = Color(0xFF00131C)

// Secondary — warm orange, the "fire palette" anchor.
val SecondaryAccent = Color(0xFFFFA45B)
val SecondaryAccentMuted = Color(0xFFE8884C)
val OnSecondaryAccent = Color(0xFF1F0E00)

// Tertiary — soft violet, used sparingly for ML / probabilistic things.
val TertiaryAccent = Color(0xFFB497FF)
val OnTertiaryAccent = Color(0xFF170A2E)

// Errors get a desaturated red — Material's default is too saturated for the
// dark backdrop and dominates the screen on the rare 4xx.
val ErrorAccent = Color(0xFFEF6B6B)
val OnErrorAccent = Color(0xFF22070A)
