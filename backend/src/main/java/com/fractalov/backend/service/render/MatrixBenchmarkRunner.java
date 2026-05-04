package com.fractalov.backend.service.render;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Runs a list of {@link BenchmarkScenario} sequentially through
 * {@link BenchmarkPipeline} and assembles a {@link MatrixResult}. Sequential
 * by design — running scenarios in parallel would have them fight each other
 * for the same {@code IntStream.parallel()} ForkJoinPool, polluting per-stage
 * timings.
 *
 * <p>Hard caps are enforced before any execution to prevent a single request
 * from saturating the server: a malicious client could otherwise pin all CPU
 * cores for hours by submitting a 1000-scenario × 100-runs matrix at the
 * highest resolution. The defaults are conservative; tuning them lives in
 * {@code application.yml} (future).
 */
@Service
public class MatrixBenchmarkRunner {

    private static final Logger log = LoggerFactory.getLogger(MatrixBenchmarkRunner.class);

    /** Hard cap on scenarios per request. ~50 lines of CSV is enough for any
     * single research figure; a real "all axes" run uses a script that calls
     * the endpoint several times. */
    public static final int MAX_SCENARIOS = 50;

    /** Hard cap on (warmup + runs) per scenario. */
    public static final int MAX_RUNS_PER_SCENARIO = 100;

    private final BenchmarkPipeline pipeline;

    public MatrixBenchmarkRunner(BenchmarkPipeline pipeline) {
        this.pipeline = pipeline;
    }

    public MatrixResult run(List<BenchmarkScenario> scenarios, int warmup, int runs, boolean includeRawSamples) {
        if (scenarios.isEmpty()) {
            throw new IllegalArgumentException("scenarios must not be empty");
        }
        if (scenarios.size() > MAX_SCENARIOS) {
            throw new IllegalArgumentException(
                    "too many scenarios: " + scenarios.size() + " > " + MAX_SCENARIOS);
        }
        if (warmup < 0 || runs < 1 || (warmup + runs) > MAX_RUNS_PER_SCENARIO) {
            throw new IllegalArgumentException(
                    "warmup+runs must be in [1, " + MAX_RUNS_PER_SCENARIO + "]");
        }

        String matrixId = UUID.randomUUID().toString();
        List<MatrixEntry> results = new ArrayList<>(scenarios.size());
        Instant outerStart = Instant.now();

        for (int i = 0; i < scenarios.size(); i++) {
            BenchmarkScenario s = scenarios.get(i);
            Instant t0 = Instant.now();
            StageBreakdown breakdown = pipeline.run(s.recipe(), warmup, runs, includeRawSamples);
            long elapsedMs = Duration.between(t0, Instant.now()).toMillis();
            results.add(new MatrixEntry(s.name(), s.tags(), breakdown));
            log.info("matrix {} [{}/{}] {}: render p50={}ms colorize p50={}ms encode p50={}ms total p50={}ms (elapsed {}ms)",
                    matrixId, i + 1, scenarios.size(), s.name(),
                    breakdown.render().p50Ms(), breakdown.colorize().p50Ms(),
                    breakdown.encode().p50Ms(), breakdown.total().p50Ms(),
                    elapsedMs);
        }

        long totalMs = Duration.between(outerStart, Instant.now()).toMillis();
        return new MatrixResult(matrixId, warmup, runs, totalMs, results);
    }
}
