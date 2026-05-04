package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.BurningShipParams;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.FractalType;
import com.fractalov.backend.service.render.kernel.BurningShipKernel;
import org.springframework.stereotype.Component;

/**
 * Burning Ship: {@code z_{n+1} = (|Re z_n| + i|Im z_n|)² + c}, {@code z_0 = 0}.
 * Non-holomorphic, so the kernel does not produce a distance-estimate field —
 * the {@link EscapeTimeEngine} will return a {@link FieldStack} with
 * {@code distanceMap = null} regardless of the recipe's colour mode.
 */
@Component
public class BurningShipRenderer implements FractalRenderer {

    private final EscapeTimeEngine engine;

    public BurningShipRenderer(EscapeTimeEngine engine) {
        this.engine = engine;
    }

    @Override
    public FractalType supports() {
        return FractalType.BURNING_SHIP;
    }

    @Override
    public RenderResult render(FractalRecipe recipe) {
        if (!(recipe.params() instanceof BurningShipParams p)) {
            throw new RenderException("BurningShipRenderer requires BurningShipParams");
        }
        BurningShipKernel kernel = new BurningShipKernel(p);
        FieldStack stack = engine.sweep(recipe.viewport(), recipe.renderSettings(), kernel, p.maxIter());
        return RenderResult.fromFieldStack(stack);
    }
}
