package com.fractalov.backend.api;

import io.zonky.test.db.AutoConfigureEmbeddedDatabase;
import io.zonky.test.db.AutoConfigureEmbeddedDatabase.DatabaseProvider;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Wire-level coverage for the matrix benchmark endpoints. Test profile sets
 * {@code app.bench.enabled=true} so the controller bean is registered.
 */
@SpringBootTest
@AutoConfigureMockMvc
@AutoConfigureEmbeddedDatabase(provider = DatabaseProvider.ZONKY)
class BenchmarkMatrixControllerTest {

    @Autowired
    MockMvc mvc;

    @Test
    void scenariosCatalogReturnsPresetsAndCollections() throws Exception {
        mvc.perform(get("/bench/scenarios"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.presets", notNullValue()))
                .andExpect(jsonPath("$.collections[0]", is("research")))
                .andExpect(jsonPath("$.collections[1]", is("full")));
    }

    @Test
    void matrixFromPresetsResearchEndToEnd() throws Exception {
        // Skip raw samples — irrelevant for shape verification and keeps the
        // response payload small.
        String body = """
                {"preset":"research","warmup":0,"runs":1,"includeRawSamples":false}
                """;
        mvc.perform(post("/bench/matrix-from-presets")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.matrix.matrixId", notNullValue()))
                .andExpect(jsonPath("$.matrix.warmup", is(0)))
                .andExpect(jsonPath("$.matrix.runs", is(1)))
                .andExpect(jsonPath("$.matrix.results", hasSize(greaterThanOrEqualTo(4))))
                .andExpect(jsonPath("$.matrix.results[0].scenario", notNullValue()))
                .andExpect(jsonPath("$.matrix.results[0].tags.family", notNullValue()))
                .andExpect(jsonPath("$.matrix.results[0].breakdown.render.runs", is(1)))
                .andExpect(jsonPath("$.matrix.results[0].breakdown.colorize", notNullValue()))
                .andExpect(jsonPath("$.matrix.results[0].breakdown.encode", notNullValue()))
                .andExpect(jsonPath("$.matrix.results[0].breakdown.total", notNullValue()));
    }

    @Test
    void matrixWithExplicitScenarios() throws Exception {
        String body = """
                {
                  "scenarios": [
                    {
                      "name": "tiny-mandel",
                      "recipe": {
                        "viewport": {"xMin":-2.0,"xMax":1.0,"yMin":-1.2,"yMax":1.2},
                        "renderSettings": {"widthPx":32,"heightPx":32},
                        "colorSettings": {"paletteName":"fire"},
                        "fractalType": "mandelbrot",
                        "params": {"maxIter":40,"escapeRadius":2.0,"smoothing":false}
                      },
                      "tags": {"axis":"smoke"}
                    }
                  ],
                  "warmup": 0,
                  "runs": 1,
                  "includeRawSamples": false
                }
                """;
        mvc.perform(post("/bench/matrix")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.matrix.results", hasSize(1)))
                .andExpect(jsonPath("$.matrix.results[0].scenario", is("tiny-mandel")))
                .andExpect(jsonPath("$.matrix.results[0].tags.axis", is("smoke")));
    }

    @Test
    void unknownPresetCollectionRejected() throws Exception {
        String body = """
                {"preset":"bogus","warmup":0,"runs":1,"includeRawSamples":false}
                """;
        // The runner throws IllegalArgumentException; ApiExceptionHandler maps it
        // to 400.
        mvc.perform(post("/bench/matrix-from-presets")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isBadRequest());
    }
}
