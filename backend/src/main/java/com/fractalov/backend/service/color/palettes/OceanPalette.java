package com.fractalov.backend.service.color.palettes;

import com.fractalov.backend.service.color.Palette;
import org.springframework.stereotype.Component;

@Component
public class OceanPalette implements Palette {

    @Override
    public String name() {
        return "ocean";
    }

    @Override
    public int argbAt(double t) {
        double clamped = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
        // deep navy -> teal -> cyan -> pale
        int r, g, b;
        if (clamped < 0.33) {
            double k = clamped / 0.33;
            r = 0;
            g = (int) Math.round(k * 80);
            b = 40 + (int) Math.round(k * 110);
        } else if (clamped < 0.66) {
            double k = (clamped - 0.33) / 0.33;
            r = 0;
            g = 80 + (int) Math.round(k * 120);
            b = 150 + (int) Math.round(k * 80);
        } else {
            double k = (clamped - 0.66) / 0.34;
            r = (int) Math.round(k * 220);
            g = 200 + (int) Math.round(k * 55);
            b = 230 + (int) Math.round(k * 25);
        }
        return (0xFF << 24) | (r << 16) | (g << 8) | b;
    }
}
