package com.fractalov.backend.api.dto;

import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.service.render.BenchmarkScenario;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

import java.util.List;
import java.util.Map;

public final class BenchmarkRequests {
    private BenchmarkRequests() {}

    /** {@code POST /bench/compare}: scalar vs vector kernel head-to-head. */
    public record CompareRequest(
            @NotNull @Valid FractalRecipe recipe,
            @Min(0) @Max(50) int warmup,
            @Min(1) @Max(100) int runs
    ) {}

    /** {@code POST /bench/matrix}: explicit list of scenarios. The caller is
     * responsible for unique names; collisions don't error but downstream
     * CSV pivots become ambiguous. */
    public record MatrixRequest(
            @NotEmpty
            @Size(max = 50)
            @Valid
            List<ScenarioSpec> scenarios,

            @Min(0) @Max(50) int warmup,
            @Min(1) @Max(100) int runs,

            /** When true, response carries raw per-run sample arrays.
             * Default off — for a 50-scenario matrix that doubles payload
             * weight without adding information for percentile-only consumers. */
            boolean includeRawSamples
    ) {}

    /** {@code POST /bench/matrix-from-presets}: caller picks preset names from
     * {@code GET /bench/scenarios}. Convenience over MatrixRequest for the
     * common research-script flow. */
    public record MatrixFromPresetsRequest(
            /** Specific scenario names. When null/empty, {@code preset} below
             * is used to choose a built-in collection. */
            List<String> names,

            /** Built-in collection: {@code "research"} (curated subset) or
             * {@code "full"} (Cartesian product). Ignored when {@code names}
             * is non-empty. */
            String preset,

            @Min(0) @Max(50) int warmup,
            @Min(1) @Max(100) int runs,
            boolean includeRawSamples
    ) {}

    /** Item of {@link MatrixRequest#scenarios()}. */
    public record ScenarioSpec(
            @NotNull String name,
            @NotNull @Valid FractalRecipe recipe,
            Map<String, String> tags
    ) {
        public BenchmarkScenario toScenario() {
            return new BenchmarkScenario(
                    name,
                    recipe,
                    tags == null ? Map.of() : Map.copyOf(tags)
            );
        }
    }
}
