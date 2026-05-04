package com.example.myapplication.domain

/**
 * Hand-picked starter recipes — one per family. They match the dataset
 * generator's "around the classic window" sampling so the user sees something
 * recognisable on first launch.
 */
object Presets {

    fun mandelbrot(): FractalRecipe = FractalRecipe(
        viewport = Viewport(xMin = -2.0, xMax = 1.0, yMin = -1.2, yMax = 1.2),
        renderSettings = RenderSettings(widthPx = 512, heightPx = 512),
        colorSettings = ColorSettings(paletteName = "fire", mode = "linear"),
        fractalType = "mandelbrot",
        params = FractalParams.Mandelbrot(maxIter = 200, escapeRadius = 2.0, smoothing = true),
    )

    fun julia(): FractalRecipe = FractalRecipe(
        viewport = Viewport(xMin = -1.5, xMax = 1.5, yMin = -1.5, yMax = 1.5),
        renderSettings = RenderSettings(widthPx = 512, heightPx = 512),
        colorSettings = ColorSettings(paletteName = "ocean", mode = "histogram"),
        fractalType = "julia",
        params = FractalParams.Julia(
            cRe = -0.7, cIm = 0.27015,
            maxIter = 200, escapeRadius = 2.0, smoothing = true,
        ),
    )

    fun burningShip(): FractalRecipe = FractalRecipe(
        viewport = Viewport(xMin = -2.0, xMax = 1.5, yMin = -2.0, yMax = 1.0),
        renderSettings = RenderSettings(widthPx = 512, heightPx = 512),
        colorSettings = ColorSettings(paletteName = "fire", mode = "linear"),
        fractalType = "burning_ship",
        params = FractalParams.BurningShip(maxIter = 200, escapeRadius = 2.0, smoothing = true),
    )

    fun multibrot(): FractalRecipe = FractalRecipe(
        viewport = Viewport(xMin = -1.5, xMax = 1.5, yMin = -1.5, yMax = 1.5),
        renderSettings = RenderSettings(widthPx = 512, heightPx = 512),
        colorSettings = ColorSettings(paletteName = "rainbow_cyclic", mode = "histogram"),
        fractalType = "multibrot",
        params = FractalParams.Multibrot(exponent = 5, maxIter = 200, escapeRadius = 2.0, smoothing = true),
    )

    // Short labels for the home selector — long names ("Burning Ship",
    // "Multibrot N=5") truncated with ellipsis at 4-up on phone widths,
    // and the coloured dot already encodes which family is which. Kept
    // recognisable by humans without the full noun phrase.
    val ALL: List<FamilyPreset> = listOf(
        FamilyPreset("mandelbrot", "Mandel", ::mandelbrot),
        FamilyPreset("julia", "Julia", ::julia),
        FamilyPreset("burning_ship", "Burning", ::burningShip),
        FamilyPreset("multibrot", "Multi", ::multibrot),
    )
}

data class FamilyPreset(
    val key: String,
    val displayName: String,
    val recipe: () -> FractalRecipe,
)
