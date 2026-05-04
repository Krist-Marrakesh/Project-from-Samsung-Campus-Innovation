package com.fractalov.backend.service.render;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class StageStatsTest {

    @Test
    void emptySamplesProduceZeroedStats() {
        StageStats s = StageStats.from("render", List.of(), false);
        assertEquals(0, s.runs());
        assertEquals(0, s.minMs());
        assertEquals(0, s.p99Ms());
    }

    @Test
    void rangeStatsAreCorrect() {
        StageStats s = StageStats.from("render",
                List.of(10L, 20L, 30L, 40L, 50L), false);
        assertEquals("render", s.stage());
        assertEquals(5, s.runs());
        assertEquals(10L, s.minMs());
        assertEquals(50L, s.maxMs());
        assertEquals(30L, s.meanMs());          // (10+20+30+40+50)/5 = 30
        assertEquals(30L, s.p50Ms());           // ceil(0.5*5)-1 = 2 → sorted[2] = 30
        assertEquals(50L, s.p90Ms());           // ceil(0.9*5)-1 = 4 → sorted[4] = 50
        assertEquals(50L, s.p99Ms());
    }

    @Test
    void unsortedSamplesAreSortedInternally() {
        StageStats s = StageStats.from("colorize",
                List.of(50L, 10L, 30L, 20L, 40L), false);
        assertEquals(10L, s.minMs());
        assertEquals(50L, s.maxMs());
        assertEquals(30L, s.p50Ms());
    }

    @Test
    void includeRawSamplesControlsPayload() {
        StageStats withRaw = StageStats.from("encode",
                List.of(1L, 2L, 3L), true);
        assertNotNull(withRaw.samplesMs());
        assertEquals(3, withRaw.samplesMs().size());

        StageStats withoutRaw = StageStats.from("encode",
                List.of(1L, 2L, 3L), false);
        assertNull(withoutRaw.samplesMs());
    }

    @Test
    void singleSampleAllPercentilesEqual() {
        StageStats s = StageStats.from("total", List.of(42L), false);
        assertEquals(42L, s.minMs());
        assertEquals(42L, s.maxMs());
        assertEquals(42L, s.p50Ms());
        assertEquals(42L, s.p99Ms());
    }
}
