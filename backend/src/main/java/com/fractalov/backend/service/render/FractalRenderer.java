package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.FractalType;

public interface FractalRenderer {
    FractalType supports();

    RenderResult render(FractalRecipe recipe);
}
