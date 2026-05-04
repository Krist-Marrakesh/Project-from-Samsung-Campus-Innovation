package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.BurningShipParams;
import com.fractalov.backend.dto.ColorMode;
import com.fractalov.backend.dto.ColorSettings;
import com.fractalov.backend.dto.FractalParams;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.JuliaParams;
import com.fractalov.backend.dto.MandelbrotParams;
import com.fractalov.backend.dto.MultibrotParams;
import com.fractalov.backend.dto.RenderSettings;
import com.fractalov.backend.dto.Viewport;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;

/**
 * Generates the benchmark matrix that the research report builds graphs on.
 *
 * <h3>Axes</h3>
 * <ul>
 *   <li><b>Family</b> — mandelbrot, julia, burning_ship, multibrot. Burning Ship
 *       skips the DE colour mode (kernel doesn't support it; see
 *       {@link com.fractalov.backend.service.render.kernel.BurningShipKernel}).</li>
 *   <li><b>Resolution</b> — 256, 512, 1024, 2048. Stops short of 4096 because
 *       a multibrot run there is north of 10 seconds and the matrix becomes
 *       unwieldy; the report can probe 4096 separately.</li>
 *   <li><b>maxIter</b> — 100, 250, 500, 1000.</li>
 *   <li><b>SSAA</b> — 1, 2 (skipping 3 because the cost dominates without
 *       teaching us something new).</li>
 *   <li><b>Colour mode</b> — linear, histogram, distance_estimate.</li>
 * </ul>
 *
 * The full Cartesian product is large (4 × 4 × 4 × 2 × 3 = 384 scenarios)
 * which would be a multi-hour run. {@link #researchPresets()} returns a
 * pre-curated subset that fits in ~10 minutes on a laptop and still covers
 * every axis at least twice. {@link #fullMatrix()} builds the full grid for
 * users who want it.
 *
 * <h3>Why the curated subset</h3>
 * Each axis is studied with the others held fixed, so reading the resulting
 * CSV does not require multivariate untangling. Concretely the subset
 * sweeps:
 * <ul>
 *   <li><i>Resolution sweep</i> — mandelbrot, maxIter=500, SSAA=1, linear,
 *       at all four resolutions.</li>
 *   <li><i>maxIter sweep</i> — mandelbrot, 1024², SSAA=1, linear, at all four
 *       iteration budgets.</li>
 *   <li><i>SSAA sweep</i> — mandelbrot, 1024², maxIter=500, linear, SSAA=1
 *       and 2.</li>
 *   <li><i>Colour mode sweep</i> — mandelbrot, 1024², maxIter=500, SSAA=1,
 *       at all three colour modes.</li>
 *   <li><i>Family sweep</i> — 1024², maxIter=500, SSAA=1, linear, across
 *       all four families.</li>
 * </ul>
 * The "anchor point" mandelbrot/1024²/iter500/SSAA1/linear appears in every
 * sweep so the report can cross-reference its baseline number.
 */
@Component
public class BenchmarkPresets {

    private static final String DEFAULT_PALETTE = "fire";

    private static final Viewport MANDELBROT_VIEW =
            new Viewport(-2.0, 1.0, -1.2, 1.2);
    private static final Viewport JULIA_VIEW =
            new Viewport(-1.5, 1.5, -1.5, 1.5);
    private static final Viewport BURNINGSHIP_VIEW =
            new Viewport(-2.0, 1.5, -2.0, 1.0);
    private static final Viewport MULTIBROT_VIEW =
            new Viewport(-1.5, 1.5, -1.5, 1.5);

    private static final double JULIA_C_RE = -0.7;
    private static final double JULIA_C_IM = 0.27015;

    private static final List<Integer> RESOLUTION_AXIS = List.of(256, 512, 1024, 2048);
    private static final List<Integer> MAXITER_AXIS = List.of(100, 250, 500, 1000);
    private static final List<Integer> SSAA_AXIS = List.of(1, 2);
    private static final List<ColorMode> COLOR_AXIS =
            List.of(ColorMode.LINEAR, ColorMode.HISTOGRAM, ColorMode.DISTANCE_ESTIMATE);
    private static final List<String> FAMILY_AXIS =
            List.of("mandelbrot", "julia", "burning_ship", "multibrot");

    /** Curated subset: ~13 scenarios, each axis swept independently. */
    public List<BenchmarkScenario> researchPresets() {
        Map<String, BenchmarkScenario> out = new LinkedHashMap<>();
        // Resolution sweep (mandelbrot, iter=500, ssaa=1, linear)
        for (int res : RESOLUTION_AXIS) {
            put(out, build("mandelbrot", res, 500, 1, ColorMode.LINEAR));
        }
        // maxIter sweep (mandelbrot, 1024, ssaa=1, linear)
        for (int it : MAXITER_AXIS) {
            put(out, build("mandelbrot", 1024, it, 1, ColorMode.LINEAR));
        }
        // SSAA sweep (mandelbrot, 1024, iter=500, linear)
        for (int ssaa : SSAA_AXIS) {
            put(out, build("mandelbrot", 1024, 500, ssaa, ColorMode.LINEAR));
        }
        // Colour mode sweep (mandelbrot, 1024, iter=500, ssaa=1)
        for (ColorMode m : COLOR_AXIS) {
            put(out, build("mandelbrot", 1024, 500, 1, m));
        }
        // Family sweep (1024, iter=500, ssaa=1, linear)
        for (String fam : FAMILY_AXIS) {
            put(out, build(fam, 1024, 500, 1, ColorMode.LINEAR));
        }
        return new ArrayList<>(out.values());
    }

    /** Full Cartesian product across every axis. Skips DE for burning_ship
     * (unsupported) and skips name collisions automatically. */
    public List<BenchmarkScenario> fullMatrix() {
        Map<String, BenchmarkScenario> out = new LinkedHashMap<>();
        for (String fam : FAMILY_AXIS) {
            for (int res : RESOLUTION_AXIS) {
                for (int it : MAXITER_AXIS) {
                    for (int ssaa : SSAA_AXIS) {
                        for (ColorMode m : COLOR_AXIS) {
                            if (m == ColorMode.DISTANCE_ESTIMATE && "burning_ship".equals(fam)) continue;
                            put(out, build(fam, res, it, ssaa, m));
                        }
                    }
                }
            }
        }
        return new ArrayList<>(out.values());
    }

    /** Build by explicit scenario names. Unknown names are silently skipped —
     * the controller surfaces them via the response shape. */
    public List<BenchmarkScenario> byNames(Collection<String> names) {
        Map<String, BenchmarkScenario> all = new LinkedHashMap<>();
        for (BenchmarkScenario s : fullMatrix()) all.put(s.name(), s);
        List<BenchmarkScenario> out = new ArrayList<>();
        for (String n : names) {
            BenchmarkScenario s = all.get(n);
            if (s != null) out.add(s);
        }
        return out;
    }

    /** Catalogue of every preset name available — sorted, used by GET /bench/scenarios. */
    public List<String> presetNames() {
        TreeSet<String> set = new TreeSet<>();
        for (BenchmarkScenario s : fullMatrix()) set.add(s.name());
        return new ArrayList<>(set);
    }

    private static void put(Map<String, BenchmarkScenario> bag, BenchmarkScenario s) {
        bag.putIfAbsent(s.name(), s);
    }

    private static BenchmarkScenario build(String family, int res, int maxIter, int ssaa, ColorMode mode) {
        FractalParams params = paramsFor(family, maxIter);
        Viewport vp = viewportFor(family);
        FractalRecipe recipe = new FractalRecipe(
                vp,
                new RenderSettings(res, res, ssaa),
                new ColorSettings(DEFAULT_PALETTE, mode),
                params
        );
        String name = String.format(Locale.ROOT,
                "%s_res%d_iter%d_ssaa%d_%s",
                family, res, maxIter, ssaa, mode.asJson());
        Map<String, String> tags = new LinkedHashMap<>();
        tags.put("family", family);
        tags.put("widthPx", String.valueOf(res));
        tags.put("heightPx", String.valueOf(res));
        tags.put("maxIter", String.valueOf(maxIter));
        tags.put("ssaa", String.valueOf(ssaa));
        tags.put("colorMode", mode.asJson());
        tags.put("palette", DEFAULT_PALETTE);
        return new BenchmarkScenario(name, recipe, Map.copyOf(tags));
    }

    private static FractalParams paramsFor(String family, int maxIter) {
        return switch (family) {
            case "mandelbrot" -> new MandelbrotParams(maxIter, 2.0, true);
            case "julia" -> new JuliaParams(JULIA_C_RE, JULIA_C_IM, maxIter, 2.0, true);
            case "burning_ship" -> new BurningShipParams(maxIter, 2.0, true);
            case "multibrot" -> new MultibrotParams(3, maxIter, 2.0, true);
            default -> throw new IllegalArgumentException("unknown family: " + family);
        };
    }

    private static Viewport viewportFor(String family) {
        return switch (family) {
            case "mandelbrot" -> MANDELBROT_VIEW;
            case "julia" -> JULIA_VIEW;
            case "burning_ship" -> BURNINGSHIP_VIEW;
            case "multibrot" -> MULTIBROT_VIEW;
            default -> throw new IllegalArgumentException("unknown family: " + family);
        };
    }
}
