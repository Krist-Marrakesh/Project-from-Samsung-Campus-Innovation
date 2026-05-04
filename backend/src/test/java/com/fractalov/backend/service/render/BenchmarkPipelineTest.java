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

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pipeline-level coverage. Avoids @SpringBootTest by wiring the dependency
 * graph manually — keeps execution under 1 second so this stays in the unit
 * tier.
 */
class BenchmarkPipelineTest {

    private final EscapeTimeEngine engine = new EscapeTimeEngine();
    private final MandelbrotRenderer mandel = new MandelbrotRenderer(engine);
    private final RenderDispatcher dispatcher = new RenderDispatcher(List.of(mandel));
    private final Colorizer colorizer = new Colorizer();
    private final PngEncoder pngEncoder = new PngEncoder();
    private final PaletteRegistry palettes = new PaletteRegistry(List.of(
            new GrayscalePalette(), new FirePalette(), new OceanPalette(), new RainbowCyclicPalette()));
    private final BenchmarkPipeline pipeline =
            new BenchmarkPipeline(dispatcher, colorizer, pngEncoder, palettes);

    private FractalRecipe smallMandel() {
        return new FractalRecipe(
                new Viewport(-2.0, 1.0, -1.2, 1.2),
                new RenderSettings(64, 64),
                new ColorSettings("fire", ColorMode.LINEAR),
                new MandelbrotParams(50, 2.0, false));
    }

    @Test
    void allFourStagesPopulated() {
        StageBreakdown b = pipeline.run(smallMandel(), 1, 3, true);
        assertAll(
                () -> assertNotNull(b.render()),
                () -> assertNotNull(b.colorize()),
                () -> assertNotNull(b.encode()),
                () -> assertNotNull(b.total()),
                () -> assertEquals(3, b.render().runs()),
                () -> assertEquals(3, b.colorize().runs()),
                () -> assertEquals(3, b.encode().runs()),
                () -> assertEquals(3, b.total().runs())
        );
    }

    @Test
    void totalIsAtLeastSumOfStagesMinusJitter() {
        // Total wall-clock should approximately equal render + colorize + encode.
        // Allow a one-millisecond fudge per stage for nano→milli truncation
        // accumulating in the unfavourable direction.
        StageBreakdown b = pipeline.run(smallMandel(), 1, 5, false);
        long perStageSum = b.render().p50Ms() + b.colorize().p50Ms() + b.encode().p50Ms();
        long totalP50 = b.total().p50Ms();
        assertTrue(totalP50 >= perStageSum - 4,
                "p50 total " + totalP50 + " < sum-of-stages " + perStageSum);
    }

    @Test
    void warmupRunsDoNotContributeSamples() {
        StageBreakdown b = pipeline.run(smallMandel(), 5, 2, true);
        assertEquals(2, b.render().samplesMs().size());
    }

    @Test
    void samplesOmittedWhenIncludeRawSamplesFalse() {
        StageBreakdown b = pipeline.run(smallMandel(), 0, 2, false);
        assertNull(b.render().samplesMs());
        assertNull(b.colorize().samplesMs());
        assertNull(b.encode().samplesMs());
        assertNull(b.total().samplesMs());
    }
}
