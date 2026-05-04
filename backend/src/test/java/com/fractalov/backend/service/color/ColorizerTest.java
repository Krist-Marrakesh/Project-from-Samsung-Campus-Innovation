package com.fractalov.backend.service.color;

import com.fractalov.backend.dto.ColorMode;
import com.fractalov.backend.service.color.palettes.GrayscalePalette;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ColorizerTest {

    private final Colorizer colorizer = new Colorizer();
    private final Palette grayscale = new GrayscalePalette();

    @Test
    void linearInSetSentinelRendersBlack() {
        double[][] map = {{-1.0}};
        BufferedImage img = colorizer.colorize(map, 100, grayscale, ColorMode.LINEAR);
        assertEquals(0xFF000000, img.getRGB(0, 0));
    }

    @Test
    void linearZeroEscapeRendersBlackOnGrayscale() {
        double[][] map = {{0.0}};
        BufferedImage img = colorizer.colorize(map, 100, grayscale, ColorMode.LINEAR);
        assertEquals(0xFF000000, img.getRGB(0, 0));
    }

    @Test
    void linearMaxEscapeRendersWhiteOnGrayscale() {
        double[][] map = {{100.0}};
        BufferedImage img = colorizer.colorize(map, 100, grayscale, ColorMode.LINEAR);
        assertEquals(0xFFFFFFFF, img.getRGB(0, 0));
    }

    @Test
    void imageDimensionsMatchEscapeMap() {
        double[][] map = new double[7][11];
        BufferedImage img = colorizer.colorize(map, 100, grayscale, ColorMode.LINEAR);
        assertEquals(11, img.getWidth());
        assertEquals(7, img.getHeight());
    }

    @Test
    void histogramStretchesClusteredEscapeValues() {
        // 10 escape pixels all at value=5 with maxIter=100 — linear gives t=0.05 (dark gray).
        // Histogram should rank them near the top of the CDF (one cohort).
        double[][] map = new double[1][10];
        for (int x = 0; x < 10; x++) map[0][x] = 5.0;

        BufferedImage linear = colorizer.colorize(map, 100, grayscale, ColorMode.LINEAR);
        BufferedImage hist = colorizer.colorize(map, 100, grayscale, ColorMode.HISTOGRAM);

        int linearGray = linear.getRGB(0, 0) & 0xFF;
        int histGray = hist.getRGB(0, 0) & 0xFF;
        assertTrue(histGray > linearGray,
                "histogram should stretch clustered values: linear=" + linearGray + " hist=" + histGray);
    }

    @Test
    void histogramMapsUniformRangeToFullSpectrum() {
        double[][] map = new double[1][256];
        for (int x = 0; x < 256; x++) map[0][x] = x;
        BufferedImage img = colorizer.colorize(map, 1000, grayscale, ColorMode.HISTOGRAM);
        Set<Integer> grays = new HashSet<>();
        for (int x = 0; x < 256; x++) {
            grays.add(img.getRGB(x, 0) & 0xFF);
        }
        // Expect a wide range of gray values produced by CDF rank.
        assertTrue(grays.size() > 200,
                "histogram should produce wide tonal range, got " + grays.size() + " distinct greys");
    }

    @Test
    void histogramPureInSetIsAllBlack() {
        double[][] map = new double[2][3];
        for (double[] row : map) {
            for (int x = 0; x < row.length; x++) row[x] = -1.0;
        }
        BufferedImage img = colorizer.colorize(map, 100, grayscale, ColorMode.HISTOGRAM);
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 3; x++) {
                assertEquals(0xFF000000, img.getRGB(x, y));
            }
        }
    }
}
