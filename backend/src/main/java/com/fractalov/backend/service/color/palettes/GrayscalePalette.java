package com.fractalov.backend.service.color.palettes;

import com.fractalov.backend.service.color.Palette;
import org.springframework.stereotype.Component;

@Component
public class GrayscalePalette implements Palette {
    @Override
    public String name() {
        return "grayscale";
    }

    @Override
    public int argbAt(double t) {
        double clamped = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
        int v = (int) Math.round(clamped * 255.0);
        return (0xFF << 24) | (v << 16) | (v << 8) | v;
    }
}
