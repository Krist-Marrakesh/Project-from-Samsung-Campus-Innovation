package com.fractalov.backend.service.color.palettes;

import com.fractalov.backend.service.color.Palette;
import org.springframework.stereotype.Component;

@Component
public class FirePalette implements Palette {

    @Override
    public String name() {
        return "fire";
    }

    @Override
    public int argbAt(double t) {
        double clamped = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
        // black -> red -> orange -> yellow -> white
        int r, g, b;
        if (clamped < 0.25) {
            double k = clamped / 0.25;
            r = (int) Math.round(k * 255);
            g = 0;
            b = 0;
        } else if (clamped < 0.5) {
            double k = (clamped - 0.25) / 0.25;
            r = 255;
            g = (int) Math.round(k * 128);
            b = 0;
        } else if (clamped < 0.75) {
            double k = (clamped - 0.5) / 0.25;
            r = 255;
            g = 128 + (int) Math.round(k * 127);
            b = 0;
        } else {
            double k = (clamped - 0.75) / 0.25;
            r = 255;
            g = 255;
            b = (int) Math.round(k * 255);
        }
        return (0xFF << 24) | (r << 16) | (g << 8) | b;
    }
}
