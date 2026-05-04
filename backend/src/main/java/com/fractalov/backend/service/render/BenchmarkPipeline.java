package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.service.color.Colorizer;
import com.fractalov.backend.service.color.Palette;
import com.fractalov.backend.service.color.PaletteRegistry;
import com.fractalov.backend.service.image.PngEncoder;
import org.springframework.stereotype.Service;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Drives a single recipe through the full {@code render → colorize → encode}
 * pipeline {@code warmup + runs} times and returns a per-stage
 * {@link StageBreakdown}.
 *
 * <h3>Why full pipeline (not just render)</h3>
 * The original {@code BenchmarkService} only times the math kernel. That answer
 * is fine when comparing two implementations of the same kernel (Stage 9's
 * scalar vs vector compare). It is the wrong answer for a research-axis
 * question of the form "is the bottleneck math, color, or encode?" — and that
 * is precisely the question the report needs to answer at multiple
 * resolutions and modes. So this harness times all three stages plus the
 * total wall-clock.
 *
 * <h3>Warmup matters</h3>
 * HotSpot's tiered compiler does not stabilise the inlining of
 * {@code IntStream.parallel().forEach} + the per-pixel inner loop until a
 * handful of iterations. With zero warmup the first run is routinely 5-10×
 * slower than steady-state, which would skew every percentile we compute.
 * Warmup runs do <em>not</em> contribute samples — they only condition the JIT.
 *
 * <h3>No persistence</h3>
 * Each run produces a {@link BufferedImage} and a base64 string that are
 * immediately discarded. We deliberately do not write to disk: filesystem
 * timing is OS-dependent, kernel-cache-dependent, and is exactly what the
 * production async-render path absorbs separately.
 */
@Service
public class BenchmarkPipeline {

    private final RenderDispatcher dispatcher;
    private final Colorizer colorizer;
    private final PngEncoder pngEncoder;
    private final PaletteRegistry palettes;

    public BenchmarkPipeline(RenderDispatcher dispatcher,
                             Colorizer colorizer,
                             PngEncoder pngEncoder,
                             PaletteRegistry palettes) {
        this.dispatcher = dispatcher;
        this.colorizer = colorizer;
        this.pngEncoder = pngEncoder;
        this.palettes = palettes;
    }

    public StageBreakdown run(FractalRecipe recipe, int warmup, int runs, boolean includeRawSamples) {
        for (int i = 0; i < warmup; i++) {
            executeOnce(recipe);
        }

        List<Long> renderSamples = new ArrayList<>(runs);
        List<Long> colorizeSamples = new ArrayList<>(runs);
        List<Long> encodeSamples = new ArrayList<>(runs);
        List<Long> totalSamples = new ArrayList<>(runs);

        for (int i = 0; i < runs; i++) {
            Timing t = executeOnce(recipe);
            renderSamples.add(t.renderMs);
            colorizeSamples.add(t.colorizeMs);
            encodeSamples.add(t.encodeMs);
            totalSamples.add(t.totalMs);
        }

        return new StageBreakdown(
                StageStats.from("render", renderSamples, includeRawSamples),
                StageStats.from("colorize", colorizeSamples, includeRawSamples),
                StageStats.from("encode", encodeSamples, includeRawSamples),
                StageStats.from("total", totalSamples, includeRawSamples)
        );
    }

    /**
     * One full render + colorize + encode pass. Mirrors the production
     * {@code RenderController} call shape so timings are representative of a
     * real user request, minus the controller envelope and Jackson serialisation
     * (those are negligible vs. the math).
     */
    private Timing executeOnce(FractalRecipe recipe) {
        Palette palette = palettes.require(recipe.colorSettings().paletteName());

        long t0 = System.nanoTime();
        RenderResult result = dispatcher.dispatch(recipe);
        long renderEnd = System.nanoTime();

        BufferedImage image = colorizer.colorize(
                result, recipe.viewport(), palette, recipe.colorSettings().effectiveMode());
        long colorizeEnd = System.nanoTime();

        // We invoke the PNG encoder for its real cost, but we discard the result
        // immediately. JIT will keep the call live because of the side effect on
        // the static ImageIO writer pool — no need for a Blackhole-style sink.
        @SuppressWarnings("unused")
        String b64 = pngEncoder.encodeToBase64(image);
        long encodeEnd = System.nanoTime();

        return new Timing(
                (renderEnd - t0) / 1_000_000L,
                (colorizeEnd - renderEnd) / 1_000_000L,
                (encodeEnd - colorizeEnd) / 1_000_000L,
                (encodeEnd - t0) / 1_000_000L
        );
    }

    private record Timing(long renderMs, long colorizeMs, long encodeMs, long totalMs) {}
}
