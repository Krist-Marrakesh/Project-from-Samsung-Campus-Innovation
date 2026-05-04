package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.ColorMode;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BenchmarkPresetsTest {

    private final BenchmarkPresets presets = new BenchmarkPresets();

    @Test
    void researchPresetsAreNonEmptyAndUniquelyNamed() {
        List<BenchmarkScenario> r = presets.researchPresets();
        assertFalse(r.isEmpty());
        Set<String> names = new HashSet<>();
        for (BenchmarkScenario s : r) names.add(s.name());
        assertEquals(r.size(), names.size(), "preset names must be unique");
    }

    @Test
    void researchPresetsCoverEachAxis() {
        List<BenchmarkScenario> r = presets.researchPresets();
        Set<String> families = new HashSet<>();
        Set<String> resolutions = new HashSet<>();
        Set<String> ssaa = new HashSet<>();
        Set<String> modes = new HashSet<>();
        Set<String> iters = new HashSet<>();
        for (BenchmarkScenario s : r) {
            families.add(s.tags().get("family"));
            resolutions.add(s.tags().get("widthPx"));
            ssaa.add(s.tags().get("ssaa"));
            modes.add(s.tags().get("colorMode"));
            iters.add(s.tags().get("maxIter"));
        }
        assertTrue(families.size() >= 4, "expect all 4 families covered, got " + families);
        assertTrue(resolutions.size() >= 4, "expect 4+ resolutions, got " + resolutions);
        assertTrue(ssaa.size() >= 2, "expect both SSAA values, got " + ssaa);
        assertTrue(modes.size() >= 3, "expect 3 colour modes, got " + modes);
        assertTrue(iters.size() >= 4, "expect 4 maxIter values, got " + iters);
    }

    @Test
    void fullMatrixSkipsDeForBurningShip() {
        List<BenchmarkScenario> all = presets.fullMatrix();
        for (BenchmarkScenario s : all) {
            if ("burning_ship".equals(s.tags().get("family"))) {
                assertFalse(ColorMode.DISTANCE_ESTIMATE.asJson()
                        .equals(s.tags().get("colorMode")),
                        "burning_ship + DE should be skipped: " + s.name());
            }
        }
    }

    @Test
    void byNamesReturnsRequestedScenariosOnly() {
        List<String> wanted = List.of(
                "mandelbrot_res1024_iter500_ssaa1_linear",
                "julia_res1024_iter500_ssaa1_linear");
        List<BenchmarkScenario> got = presets.byNames(wanted);
        assertEquals(2, got.size());
        assertEquals(wanted.get(0), got.get(0).name());
        assertEquals(wanted.get(1), got.get(1).name());
    }

    @Test
    void byNamesSilentlyDropsUnknown() {
        List<BenchmarkScenario> got = presets.byNames(List.of("nope_no_such_thing"));
        assertTrue(got.isEmpty());
    }

    @Test
    void presetNamesContainsKnownAnchorPoint() {
        // Every research sweep includes the "anchor" point that all axes
        // cross-reference. If the anchor name drifts the report's baseline
        // breaks, so we assert it stays stable here.
        List<String> names = presets.presetNames();
        assertTrue(names.contains("mandelbrot_res1024_iter500_ssaa1_linear"),
                "anchor scenario must remain in the catalogue");
    }
}
