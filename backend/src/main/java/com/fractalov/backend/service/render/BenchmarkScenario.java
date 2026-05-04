package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.FractalRecipe;

import java.util.Map;

/**
 * One point in a benchmark matrix: a recipe to render plus structured tags
 * that downstream tooling (CSV pivots, plots) groups on. The {@code name}
 * uniquely identifies the scenario in a result row; tags carry the axes
 * that the recipe encodes (family, resolution, maxIter, ssaa, colorMode, …)
 * so a CSV consumer doesn't have to re-parse the recipe to chart by axis.
 */
public record BenchmarkScenario(
        String name,
        FractalRecipe recipe,
        Map<String, String> tags
) {}
