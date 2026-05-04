package com.example.myapplication.ui

/**
 * Routes encoded as constants, with a single helper for routes carrying
 * arguments. Keeping it data-class-free is deliberate — Navigation Compose's
 * builder API takes plain string templates, and ``{arg}`` interpolation is
 * the smallest readable thing.
 *
 * One route accepts a JSON-encoded recipe (variations). We URL-encode it on
 * entry; this avoids a session-state singleton or a separate "passing recipes"
 * channel.
 */
object Routes {
    const val HOME = "home"
    const val ML = "ml"
    const val COMPARISON = "comparison"

    private const val VARIATIONS_BASE = "variations"
    private const val ARG_RECIPE_JSON = "recipeJson"
    const val VARIATIONS_PATTERN = "$VARIATIONS_BASE?$ARG_RECIPE_JSON={$ARG_RECIPE_JSON}"
    val VARIATIONS_ARG = ARG_RECIPE_JSON

    fun variations(recipeJson: String): String {
        val encoded = java.net.URLEncoder.encode(recipeJson, Charsets.UTF_8)
        return "$VARIATIONS_BASE?$ARG_RECIPE_JSON=$encoded"
    }
}
