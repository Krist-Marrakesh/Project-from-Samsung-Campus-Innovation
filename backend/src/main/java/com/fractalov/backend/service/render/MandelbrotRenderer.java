package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.ColorMode;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.FractalType;
import com.fractalov.backend.dto.MandelbrotParams;
import com.fractalov.backend.service.render.kernel.MandelbrotKernel;
import org.springframework.stereotype.Component;

/**
 * Family-specific entry point: builds a {@link MandelbrotKernel} closed over
 * the recipe's parameters and runs it through the shared
 * {@link EscapeTimeEngine}. Distance estimation is only computed when the
 * recipe asks for it (colour mode {@link ColorMode#DISTANCE_ESTIMATE}); for
 * every other mode the derivative track is skipped to keep the iteration
 * loop tight.
 */
@Component
public class MandelbrotRenderer implements FractalRenderer {

    private final EscapeTimeEngine engine;

    public MandelbrotRenderer(EscapeTimeEngine engine) {
        this.engine = engine;
    }

    @Override
    public FractalType supports() {
        return FractalType.MANDELBROT;
    }

    @Override
    public RenderResult render(FractalRecipe recipe) {
        if (!(recipe.params() instanceof MandelbrotParams p)) {
            throw new RenderException("MandelbrotRenderer requires MandelbrotParams");
        }
        boolean wantDe = recipe.colorSettings().effectiveMode() == ColorMode.DISTANCE_ESTIMATE;
        MandelbrotKernel kernel = new MandelbrotKernel(p, wantDe);
        FieldStack stack = engine.sweep(recipe.viewport(), recipe.renderSettings(), kernel, p.maxIter());
        return RenderResult.fromFieldStack(stack);
    }
}
