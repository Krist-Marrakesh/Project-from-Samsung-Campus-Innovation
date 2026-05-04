package com.fractalov.backend.service.render;

import java.util.Map;

/** One row of a {@link MatrixResult}: scenario identity (name + tags) and
 * the per-stage timing breakdown. Tags are repeated here because flattening
 * to CSV downstream is much simpler when each row carries its own context. */
public record MatrixEntry(
        String scenario,
        Map<String, String> tags,
        StageBreakdown breakdown
) {}
