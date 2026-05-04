package com.fractalov.backend.api.dto;

import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.service.ml.MlSuggestion;

import java.util.List;
import java.util.Map;
import java.util.UUID;

public final class MlResponses {
    private MlResponses() {}

    public record SuggestionView(
            String family,
            double familyConfidence,
            Map<String, Double> familyDistribution,
            Double cRe,
            Double cIm,
            FractalRecipe suggestedRecipe
    ) {
        public static SuggestionView of(MlSuggestion s) {
            return new SuggestionView(
                    s.family(),
                    s.familyConfidence(),
                    s.familyDistribution(),
                    s.cRe(),
                    s.cIm(),
                    s.recipe()
            );
        }
    }

    public record RenderFromImageView(
            SuggestionView suggestion,
            UUID projectId,
            UUID recipeId,
            RenderRecordResponses.View render
    ) {}

    public record VariationsView(List<FractalRecipe> recipes) {}
}
