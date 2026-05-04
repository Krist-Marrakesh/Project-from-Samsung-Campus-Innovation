package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.FractalType;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

@Component
public class RenderDispatcher {

    private final Map<FractalType, FractalRenderer> byType;

    public RenderDispatcher(List<FractalRenderer> renderers) {
        this.byType = renderers.stream()
                .collect(Collectors.toUnmodifiableMap(FractalRenderer::supports, Function.identity()));
    }

    public RenderResult dispatch(FractalRecipe recipe) {
        FractalType type = recipe.fractalType();
        FractalRenderer renderer = byType.get(type);
        if (renderer == null) {
            throw new RenderException("No renderer registered for fractal type: " + type);
        }
        return renderer.render(recipe);
    }

    public List<FractalType> supportedTypes() {
        return byType.keySet().stream().sorted().toList();
    }
}
