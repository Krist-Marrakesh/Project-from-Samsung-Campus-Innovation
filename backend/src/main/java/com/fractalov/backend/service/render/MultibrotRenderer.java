package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.ColorMode;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.FractalType;
import com.fractalov.backend.dto.MultibrotParams;
import com.fractalov.backend.service.render.kernel.MultibrotKernel;
import org.springframework.stereotype.Component;

@Component
public class MultibrotRenderer implements FractalRenderer {

    private final EscapeTimeEngine engine;

    public MultibrotRenderer(EscapeTimeEngine engine) {
        this.engine = engine;
    }

    @Override
    public FractalType supports() {
        return FractalType.MULTIBROT;
    }

    @Override
    public RenderResult render(FractalRecipe recipe) {
        if (!(recipe.params() instanceof MultibrotParams p)) {
            throw new RenderException("MultibrotRenderer requires MultibrotParams");
        }
        boolean wantDe = recipe.colorSettings().effectiveMode() == ColorMode.DISTANCE_ESTIMATE;
        MultibrotKernel kernel = new MultibrotKernel(p, wantDe);
        FieldStack stack = engine.sweep(recipe.viewport(), recipe.renderSettings(), kernel, p.maxIter());
        return RenderResult.fromFieldStack(stack);
    }
}
