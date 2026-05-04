package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.ColorMode;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.FractalType;
import com.fractalov.backend.dto.JuliaParams;
import com.fractalov.backend.service.render.kernel.JuliaKernel;
import org.springframework.stereotype.Component;

@Component
public class JuliaRenderer implements FractalRenderer {

    private final EscapeTimeEngine engine;

    public JuliaRenderer(EscapeTimeEngine engine) {
        this.engine = engine;
    }

    @Override
    public FractalType supports() {
        return FractalType.JULIA;
    }

    @Override
    public RenderResult render(FractalRecipe recipe) {
        if (!(recipe.params() instanceof JuliaParams p)) {
            throw new RenderException("JuliaRenderer requires JuliaParams");
        }
        boolean wantDe = recipe.colorSettings().effectiveMode() == ColorMode.DISTANCE_ESTIMATE;
        JuliaKernel kernel = new JuliaKernel(p, wantDe);
        FieldStack stack = engine.sweep(recipe.viewport(), recipe.renderSettings(), kernel, p.maxIter());
        return RenderResult.fromFieldStack(stack);
    }
}
