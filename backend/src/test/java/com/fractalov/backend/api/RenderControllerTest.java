package com.fractalov.backend.api;

import io.zonky.test.db.AutoConfigureEmbeddedDatabase;
import io.zonky.test.db.AutoConfigureEmbeddedDatabase.DatabaseProvider;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@AutoConfigureEmbeddedDatabase(provider = DatabaseProvider.ZONKY)
class RenderControllerTest {

    @Autowired
    MockMvc mvc;

    private static final String MANDEL_OK = """
            {
              "recipe": {
                "viewport": {"xMin":-2.0,"xMax":1.0,"yMin":-1.2,"yMax":1.2},
                "renderSettings": {"widthPx":64,"heightPx":64},
                "colorSettings": {"paletteName":"fire"},
                "fractalType": "mandelbrot",
                "params": {"maxIter":100,"escapeRadius":2.0,"smoothing":true}
              }
            }
            """;

    private static final String JULIA_OK = """
            {
              "recipe": {
                "viewport": {"xMin":-1.5,"xMax":1.5,"yMin":-1.5,"yMax":1.5},
                "renderSettings": {"widthPx":64,"heightPx":64},
                "colorSettings": {"paletteName":"ocean"},
                "fractalType": "julia",
                "params": {"cRe":-0.7,"cIm":0.27015,"maxIter":100,"escapeRadius":2.0,"smoothing":true}
              }
            }
            """;

    private static final String BURNING_SHIP_OK = """
            {
              "recipe": {
                "viewport": {"xMin":-2.0,"xMax":1.5,"yMin":-2.0,"yMax":1.0},
                "renderSettings": {"widthPx":64,"heightPx":64},
                "colorSettings": {"paletteName":"fire"},
                "fractalType": "burning_ship",
                "params": {"maxIter":100,"escapeRadius":2.0,"smoothing":true}
              }
            }
            """;

    private static final String MULTIBROT_OK = """
            {
              "recipe": {
                "viewport": {"xMin":-1.5,"xMax":1.5,"yMin":-1.5,"yMax":1.5},
                "renderSettings": {"widthPx":64,"heightPx":64},
                "colorSettings": {"paletteName":"rainbow_cyclic"},
                "fractalType": "multibrot",
                "params": {"exponent":3,"maxIter":100,"escapeRadius":2.0,"smoothing":true}
              }
            }
            """;

    @Test
    void rendersMandelbrot() throws Exception {
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(MANDEL_OK))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status", is("ok")))
                .andExpect(jsonPath("$.format", is("png")))
                .andExpect(jsonPath("$.widthPx", is(64)))
                .andExpect(jsonPath("$.heightPx", is(64)))
                .andExpect(jsonPath("$.imageBase64", notNullValue()))
                .andExpect(jsonPath("$.requestId", notNullValue()))
                .andExpect(jsonPath("$.performance.renderMs", greaterThanOrEqualTo(0)))
                .andExpect(jsonPath("$.performance.colorizeMs", greaterThanOrEqualTo(0)))
                .andExpect(jsonPath("$.performance.encodeMs", greaterThanOrEqualTo(0)))
                .andExpect(jsonPath("$.performance.totalMs", greaterThanOrEqualTo(0)));
    }

    @Test
    void rendersJulia() throws Exception {
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(JULIA_OK))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.imageBase64", notNullValue()));
    }

    @Test
    void rendersBurningShip() throws Exception {
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(BURNING_SHIP_OK))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.imageBase64", notNullValue()));
    }

    @Test
    void rendersMultibrot() throws Exception {
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(MULTIBROT_OK))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.imageBase64", notNullValue()));
    }

    @Test
    void rendersWithSsaa() throws Exception {
        String body = MANDEL_OK.replace(
                "\"renderSettings\": {\"widthPx\":64,\"heightPx\":64}",
                "\"renderSettings\": {\"widthPx\":32,\"heightPx\":32,\"samplesPerAxis\":2}");
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.widthPx", is(32)))
                .andExpect(jsonPath("$.imageBase64", notNullValue()));
    }

    @Test
    void rendersWithHistogramColoring() throws Exception {
        String body = MANDEL_OK.replace(
                "\"colorSettings\": {\"paletteName\":\"fire\"}",
                "\"colorSettings\": {\"paletteName\":\"fire\",\"mode\":\"histogram\"}");
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.imageBase64", notNullValue()))
                .andExpect(jsonPath("$.recipeEcho.colorSettings.mode", is("histogram")));
    }

    @Test
    void rejectsSsaaAboveLimit() throws Exception {
        String body = MANDEL_OK.replace(
                "\"renderSettings\": {\"widthPx\":64,\"heightPx\":64}",
                "\"renderSettings\": {\"widthPx\":32,\"heightPx\":32,\"samplesPerAxis\":5}");
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isBadRequest());
    }

    @Test
    void rejectsInvertedViewport() throws Exception {
        String body = MANDEL_OK.replace("\"xMin\":-2.0,\"xMax\":1.0", "\"xMin\":1.0,\"xMax\":-2.0");
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", is("validation_failed")));
    }

    @Test
    void rejectsZeroMaxIter() throws Exception {
        String body = MANDEL_OK.replace("\"maxIter\":100", "\"maxIter\":0");
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", is("validation_failed")));
    }

    @Test
    void rejectsZeroWidth() throws Exception {
        String body = MANDEL_OK.replace("\"widthPx\":64", "\"widthPx\":0");
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", is("validation_failed")));
    }

    @Test
    void rejectsUnknownFractalType() throws Exception {
        String body = MANDEL_OK.replace("\"fractalType\": \"mandelbrot\"", "\"fractalType\": \"totally-unknown\"");
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", is("unknown_fractal_type")));
    }

    @Test
    void rejectsUnknownPalette() throws Exception {
        String body = MANDEL_OK.replace("\"paletteName\":\"fire\"", "\"paletteName\":\"nope-palette\"");
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", is("bad_argument")));
    }

    @Test
    void rejectsMultibrotExponentTooHigh() throws Exception {
        String body = MULTIBROT_OK.replace("\"exponent\":3", "\"exponent\":99");
        mvc.perform(post("/render").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isBadRequest());
    }

    @Test
    void validateRecipeFindsMultipleErrors() throws Exception {
        String body = """
                {
                  "recipe": {
                    "viewport": {"xMin":1.0,"xMax":-1.0,"yMin":0.0,"yMax":1.0},
                    "renderSettings": {"widthPx":0,"heightPx":64},
                    "colorSettings": {"paletteName":"nope"},
                    "fractalType": "mandelbrot",
                    "params": {"maxIter":0,"escapeRadius":2.0,"smoothing":false}
                  }
                }
                """;
        mvc.perform(post("/validate-recipe").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.valid", is(false)))
                .andExpect(jsonPath("$.errors.length()", greaterThan(0)));
    }

    @Test
    void validateRecipeAcceptsGoodInput() throws Exception {
        mvc.perform(post("/validate-recipe").contentType(MediaType.APPLICATION_JSON).content(MANDEL_OK))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.valid", is(true)));
    }
}
