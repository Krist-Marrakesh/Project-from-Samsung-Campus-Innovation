package com.fractalov.backend.service.render;

import java.util.List;

/**
 * Outcome of a {@link MatrixBenchmarkRunner#run} call: a stable identifier,
 * the warmup/runs that produced these numbers, total wall-clock for the
 * whole sweep, and one entry per scenario.
 *
 * <p>This is the JSON the {@code BenchmarkController} serialises; the field
 * shapes are part of the public contract for downstream tooling (CSV pivot
 * scripts, the research-report generator).
 */
public record MatrixResult(
        String matrixId,
        int warmup,
        int runs,
        long elapsedMs,
        List<MatrixEntry> results
) {}
