package com.fractalov.backend.service.ml;

import com.fractalov.backend.dto.FractalRecipe;

import java.util.Map;

/**
 * Wire-shape mirror of the Python ML server's {@code SuggestionResponse}.
 * Embeds a fully-formed {@link FractalRecipe} that is ready to feed into the
 * existing render pipeline — the ML service composes it on its side rather
 * than the Java backend stitching together family + c with default scaffolding.
 */
public record MlSuggestion(
        String family,
        double familyConfidence,
        Map<String, Double> familyDistribution,
        Double cRe,
        Double cIm,
        FractalRecipe recipe
) {}
