package com.fractalov.backend.service.render;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Aggregate timing statistics for one pipeline stage across {@code N} sample
 * runs. {@code samples} is optionally populated (request-controlled) so a
 * caller doing CSV/graph generation can keep raw observations, while a caller
 * doing fleet-level aggregation pays no payload weight for them.
 */
public record StageStats(
        String stage,
        int runs,
        long minMs,
        long maxMs,
        long meanMs,
        long p50Ms,
        long p90Ms,
        long p99Ms,
        List<Long> samplesMs
) {

    /**
     * Build a {@link StageStats} from raw per-run observations. Sorts the
     * samples and computes nearest-rank percentiles. {@code includeRawSamples}
     * decides whether the raw list is surfaced — large matrices set this false
     * to keep responses bounded.
     */
    public static StageStats from(String stage, List<Long> samples, boolean includeRawSamples) {
        if (samples.isEmpty()) {
            return new StageStats(stage, 0, 0, 0, 0, 0, 0, 0,
                    includeRawSamples ? List.of() : null);
        }
        List<Long> sorted = new ArrayList<>(samples);
        Collections.sort(sorted);
        long min = sorted.get(0);
        long max = sorted.get(sorted.size() - 1);
        long sum = 0;
        for (long v : sorted) sum += v;
        long mean = sum / sorted.size();
        return new StageStats(
                stage,
                sorted.size(),
                min,
                max,
                mean,
                percentile(sorted, 0.50),
                percentile(sorted, 0.90),
                percentile(sorted, 0.99),
                includeRawSamples ? List.copyOf(sorted) : null
        );
    }

    private static long percentile(List<Long> sorted, double p) {
        // Nearest-rank percentile — adequate for the small sample sizes this
        // harness uses (N ≤ 100). Linear interpolation would buy us nothing
        // because run-to-run variance dominates the percentile estimate.
        int idx = (int) Math.ceil(p * sorted.size()) - 1;
        if (idx < 0) idx = 0;
        if (idx >= sorted.size()) idx = sorted.size() - 1;
        return sorted.get(idx);
    }
}
