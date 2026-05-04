package com.fractalov.backend.service.render;

import com.fractalov.backend.api.dto.BenchmarkResponses.KernelStats;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.FractalType;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

/**
 * Stage 9 microbenchmark harness. Runs a render kernel through warmup +
 * timed iterations on the same recipe and reports min / max / mean / p50 /
 * p90 / p99 of the wall-clock {@code render} time only — color, encode and
 * everything else are excluded so the comparison stays focused on the math.
 *
 * <p>Warmup matters here: HotSpot's tiered compiler needs a few iterations
 * before it stabilises the inlining of {@code IntStream.parallel().forEach}
 * and the inner double loop, and the Vector API renderer also needs warmup
 * for HotSpot's vector intrinsic recogniser to kick in. With zero warmup the
 * first iteration is routinely 5-10× slower than steady state.
 */
@Service
public class BenchmarkService {

    private final MandelbrotRenderer scalar;
    private final MandelbrotVectorRenderer vector;

    public BenchmarkService(MandelbrotRenderer scalar, MandelbrotVectorRenderer vector) {
        this.scalar = scalar;
        this.vector = vector;
    }

    public KernelStats benchmarkScalar(FractalRecipe recipe, int warmup, int runs) {
        requireMandelbrot(recipe);
        return measure("scalar", 1, recipe, warmup, runs, scalar::render);
    }

    public KernelStats benchmarkVector(FractalRecipe recipe, int warmup, int runs) {
        requireMandelbrot(recipe);
        return measure("vector", vector.laneCount(), recipe, warmup, runs, vector::render);
    }

    private static void requireMandelbrot(FractalRecipe recipe) {
        if (recipe.fractalType() != FractalType.MANDELBROT) {
            throw new RenderException("benchmark currently only supports Mandelbrot");
        }
    }

    private static KernelStats measure(
            String name,
            int laneCount,
            FractalRecipe recipe,
            int warmup,
            int runs,
            Function<FractalRecipe, RenderResult> kernel) {
        for (int i = 0; i < warmup; i++) {
            kernel.apply(recipe);
        }
        List<Long> samples = new ArrayList<>(runs);
        for (int i = 0; i < runs; i++) {
            long start = System.nanoTime();
            kernel.apply(recipe);
            samples.add((System.nanoTime() - start) / 1_000_000L);
        }
        return summarise(name, laneCount, samples);
    }

    private static KernelStats summarise(String name, int laneCount, List<Long> samples) {
        Collections.sort(samples);
        long min = samples.get(0);
        long max = samples.get(samples.size() - 1);
        long sum = 0;
        for (long v : samples) sum += v;
        long mean = sum / samples.size();
        return new KernelStats(
                name,
                laneCount,
                samples.size(),
                min,
                max,
                mean,
                percentile(samples, 0.50),
                percentile(samples, 0.90),
                percentile(samples, 0.99),
                List.copyOf(samples)
        );
    }

    private static long percentile(List<Long> sorted, double p) {
        if (sorted.isEmpty()) return 0;
        // Nearest-rank — adequate for the small sample sizes we use (≤100).
        int idx = (int) Math.ceil(p * sorted.size()) - 1;
        if (idx < 0) idx = 0;
        if (idx >= sorted.size()) idx = sorted.size() - 1;
        return sorted.get(idx);
    }
}
