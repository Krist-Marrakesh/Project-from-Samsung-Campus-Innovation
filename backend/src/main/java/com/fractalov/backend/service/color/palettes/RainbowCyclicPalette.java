package com.fractalov.backend.service.color.palettes;

import com.fractalov.backend.service.color.Palette;
import org.springframework.stereotype.Component;

@Component
public class RainbowCyclicPalette implements Palette {

    @Override
    public String name() {
        return "rainbow_cyclic";
    }

    @Override
    public int argbAt(double t) {
        double frac = t - Math.floor(t);
        double hue = frac * 360.0;
        return hsvToArgb(hue, 0.85, 0.95);
    }

    private static int hsvToArgb(double hDeg, double s, double v) {
        double c = v * s;
        double hp = hDeg / 60.0;
        double x = c * (1.0 - Math.abs((hp % 2.0) - 1.0));
        double r1, g1, b1;
        int sector = (int) Math.floor(hp) % 6;
        if (sector < 0) sector += 6;
        switch (sector) {
            case 0 -> { r1 = c; g1 = x; b1 = 0; }
            case 1 -> { r1 = x; g1 = c; b1 = 0; }
            case 2 -> { r1 = 0; g1 = c; b1 = x; }
            case 3 -> { r1 = 0; g1 = x; b1 = c; }
            case 4 -> { r1 = x; g1 = 0; b1 = c; }
            default -> { r1 = c; g1 = 0; b1 = x; }
        }
        double m = v - c;
        int r = (int) Math.round((r1 + m) * 255.0);
        int g = (int) Math.round((g1 + m) * 255.0);
        int b = (int) Math.round((b1 + m) * 255.0);
        return (0xFF << 24) | (clamp(r) << 16) | (clamp(g) << 8) | clamp(b);
    }

    private static int clamp(int v) {
        return v < 0 ? 0 : (v > 255 ? 255 : v);
    }
}
