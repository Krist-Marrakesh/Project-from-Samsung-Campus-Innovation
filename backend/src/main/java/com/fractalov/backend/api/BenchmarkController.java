package com.fractalov.backend.api;

import com.fractalov.backend.api.dto.BenchmarkRequests;
import com.fractalov.backend.api.dto.BenchmarkResponses;
import com.fractalov.backend.api.dto.BenchmarkResponses.KernelStats;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.service.render.BenchmarkPresets;
import com.fractalov.backend.service.render.BenchmarkScenario;
import com.fractalov.backend.service.render.BenchmarkService;
import com.fractalov.backend.service.render.MatrixBenchmarkRunner;
import com.fractalov.backend.service.render.MatrixResult;
import jakarta.validation.Valid;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * Stage 9 / Slice 1 research endpoints. Two flavours:
 *
 * <ol>
 *   <li>{@code /bench/compare} — head-to-head scalar vs optimized Mandelbrot
 *       on one recipe. Existing Stage 9 entry point.</li>
 *   <li>{@code /bench/matrix*} — full pipeline (render + colorize + encode)
 *       sweep across a set of scenarios, returning per-stage percentile
 *       breakdowns. The matrix is what the research report builds graphs
 *       on.</li>
 * </ol>
 *
 * <p>Gated by {@code app.bench.enabled} ({@code FRACTALOV_BENCH_ENABLED} env
 * var). Default off — these endpoints are unauthenticated and run uncapped
 * CPU-intensive code, which is only acceptable in research / dev contexts.
 * The conditional removes the controller bean entirely when the flag is off,
 * so the routes return 404 rather than rejecting at request time.
 */
@RestController
@RequestMapping("/bench")
@ConditionalOnProperty(name = "app.bench.enabled", havingValue = "true")
public class BenchmarkController {

    private final BenchmarkService bench;
    private final MatrixBenchmarkRunner matrix;
    private final BenchmarkPresets presets;

    public BenchmarkController(BenchmarkService bench,
                               MatrixBenchmarkRunner matrix,
                               BenchmarkPresets presets) {
        this.bench = bench;
        this.matrix = matrix;
        this.presets = presets;
    }

    @PostMapping("/compare")
    public BenchmarkResponses.CompareResponse compare(
            @Valid @RequestBody BenchmarkRequests.CompareRequest req) {
        FractalRecipe recipe = req.recipe();
        KernelStats scalar = bench.benchmarkScalar(recipe, req.warmup(), req.runs());
        KernelStats vector = bench.benchmarkVector(recipe, req.warmup(), req.runs());

        // Use medians for the speedup ratio — robust to a single outlier
        // run hitting GC or thermal throttling, both common on a laptop.
        double speedup = vector.p50Ms() == 0 ? Double.POSITIVE_INFINITY
                : (double) scalar.p50Ms() / (double) vector.p50Ms();

        return new BenchmarkResponses.CompareResponse(
                recipe.renderSettings().widthPx(),
                recipe.renderSettings().heightPx(),
                ((com.fractalov.backend.dto.MandelbrotParams) recipe.params()).maxIter(),
                req.warmup(),
                scalar,
                vector,
                speedup
        );
    }

    @GetMapping("/scenarios")
    public BenchmarkResponses.ScenariosCatalog scenarios() {
        return new BenchmarkResponses.ScenariosCatalog(
                presets.presetNames(),
                List.of("research", "full")
        );
    }

    @PostMapping("/matrix")
    public BenchmarkResponses.MatrixResponse matrix(
            @Valid @RequestBody BenchmarkRequests.MatrixRequest req) {
        List<BenchmarkScenario> scenarios = req.scenarios().stream()
                .map(BenchmarkRequests.ScenarioSpec::toScenario)
                .toList();
        MatrixResult result = matrix.run(scenarios, req.warmup(), req.runs(), req.includeRawSamples());
        return new BenchmarkResponses.MatrixResponse(result);
    }

    @PostMapping("/matrix-from-presets")
    public BenchmarkResponses.MatrixResponse matrixFromPresets(
            @Valid @RequestBody BenchmarkRequests.MatrixFromPresetsRequest req) {
        List<BenchmarkScenario> scenarios;
        if (req.names() != null && !req.names().isEmpty()) {
            scenarios = presets.byNames(req.names());
            if (scenarios.isEmpty()) {
                throw new IllegalArgumentException(
                        "none of the requested preset names matched: " + req.names());
            }
        } else {
            String collection = req.preset() == null ? "research" : req.preset();
            scenarios = switch (collection) {
                case "research" -> presets.researchPresets();
                case "full" -> presets.fullMatrix();
                default -> throw new IllegalArgumentException(
                        "unknown preset collection: " + collection
                                + " (expected 'research' or 'full')");
            };
        }
        MatrixResult result = matrix.run(scenarios, req.warmup(), req.runs(), req.includeRawSamples());
        return new BenchmarkResponses.MatrixResponse(result);
    }
}
