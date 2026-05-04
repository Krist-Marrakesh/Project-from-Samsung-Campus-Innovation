package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.ColorMode;
import com.fractalov.backend.dto.ColorSettings;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.MandelbrotParams;
import com.fractalov.backend.dto.RenderSettings;
import com.fractalov.backend.dto.Viewport;
import com.fractalov.backend.service.color.Colorizer;
import com.fractalov.backend.service.color.PaletteRegistry;
import com.fractalov.backend.service.color.palettes.FirePalette;
import com.fractalov.backend.service.color.palettes.GrayscalePalette;
import com.fractalov.backend.service.color.palettes.OceanPalette;
import com.fractalov.backend.service.color.palettes.RainbowCyclicPalette;
import com.fractalov.backend.service.image.PngEncoder;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MatrixBenchmarkRunnerTest {

    private final EscapeTimeEngine engine = new EscapeTimeEngine();
    private final MandelbrotRenderer mandel = new MandelbrotRenderer(engine);
    private final RenderDispatcher dispatcher = new RenderDispatcher(List.of(mandel));
    private final Colorizer colorizer = new Colorizer();
    private final PngEncoder pngEncoder = new PngEncoder();
    private final PaletteRegistry palettes = new PaletteRegistry(List.of(
            new GrayscalePalette(), new FirePalette(), new OceanPalette(), new RainbowCyclicPalette()));
    private final BenchmarkPipeline pipeline =
            new BenchmarkPipeline(dispatcher, colorizer, pngEncoder, palettes);
    private final MatrixBenchmarkRunner runner = new MatrixBenchmarkRunner(pipeline);

    private BenchmarkScenario tinyMandel(String name) {
        FractalRecipe recipe = new FractalRecipe(
                new Viewport(-2.0, 1.0, -1.2, 1.2),
                new RenderSettings(32, 32),
                new ColorSettings("fire", ColorMode.LINEAR),
                new MandelbrotParams(50, 2.0, false));
        return new BenchmarkScenario(
                name,
                recipe,
                Map.of("family", "mandelbrot", "widthPx", "32"));
    }

    @Test
    void runsOneEntryPerScenarioAndPreservesOrder() {
        List<BenchmarkScenario> in = List.of(
                tinyMandel("first"),
                tinyMandel("second"),
                tinyMandel("third"));
        MatrixResult r = runner.run(in, 0, 1, false);

        assertNotNull(r.matrixId());
        assertEquals(3, r.results().size());
        assertEquals("first", r.results().get(0).scenario());
        assertEquals("second", r.results().get(1).scenario());
        assertEquals("third", r.results().get(2).scenario());
        assertTrue(r.elapsedMs() >= 0);
    }

    @Test
    void resultsCarryTagsForCsvPivot() {
        MatrixResult r = runner.run(List.of(tinyMandel("only")), 0, 1, false);
        Map<String, String> tags = r.results().get(0).tags();
        assertEquals("mandelbrot", tags.get("family"));
        assertEquals("32", tags.get("widthPx"));
    }

    @Test
    void emptyScenariosRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> runner.run(List.of(), 0, 1, false));
    }

    @Test
    void tooManyScenariosRejected() {
        List<BenchmarkScenario> too =
                new ArrayList<>(MatrixBenchmarkRunner.MAX_SCENARIOS + 1);
        for (int i = 0; i <= MatrixBenchmarkRunner.MAX_SCENARIOS; i++) {
            too.add(tinyMandel("s" + i));
        }
        assertThrows(IllegalArgumentException.class,
                () -> runner.run(too, 0, 1, false));
    }

    @Test
    void warmupPlusRunsCappedAt100() {
        assertThrows(IllegalArgumentException.class,
                () -> runner.run(List.of(tinyMandel("only")), 50, 51, false));
    }
}
