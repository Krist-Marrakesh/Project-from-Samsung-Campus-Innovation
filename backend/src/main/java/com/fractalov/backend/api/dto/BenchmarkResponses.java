package com.fractalov.backend.api.dto;

import com.fractalov.backend.service.render.MatrixResult;

import java.util.List;

public final class BenchmarkResponses {
    private BenchmarkResponses() {}

    /** Stats for one kernel under {@code /bench/compare}. */
    public record KernelStats(
            String kernel,
            int laneCount,
            int runs,
            long minMs,
            long maxMs,
            long meanMs,
            long p50Ms,
            long p90Ms,
            long p99Ms,
            List<Long> samplesMs
    ) {}

    public record CompareResponse(
            int widthPx,
            int heightPx,
            int maxIter,
            int warmup,
            KernelStats scalar,
            KernelStats vector,
            double speedup
    ) {}

    /** {@code GET /bench/scenarios}: catalogue of built-in preset names. */
    public record ScenariosCatalog(List<String> presets, List<String> collections) {}

    /** Result envelope for {@code /bench/matrix*}. The {@link MatrixResult}
     * is already JSON-shaped; this wrapper exists so future fields (host
     * metadata, build hash) can join the response without breaking
     * MatrixResult's narrower computation-only contract. */
    public record MatrixResponse(MatrixResult matrix) {}
}
