package com.fractalov.backend.service.render;

/**
 * Per-stage timing decomposition for one benchmarked recipe. The four stages
 * map onto the production render path: math kernel → colorize → PNG encode,
 * with {@code total} aggregating wall-clock time including dispatch overhead.
 *
 * <p>Decomposition is the whole point — knowing total time alone does not let
 * you say where the bottleneck is. With per-stage stats the research report
 * can show, for example, that at 4096² resolution PNG encode dominates total
 * time, while at 1024² it's the math kernel.
 */
public record StageBreakdown(
        StageStats render,
        StageStats colorize,
        StageStats encode,
        StageStats total
) {}
