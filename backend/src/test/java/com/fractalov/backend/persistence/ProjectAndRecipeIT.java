package com.fractalov.backend.persistence;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.zonky.test.db.AutoConfigureEmbeddedDatabase;
import io.zonky.test.db.AutoConfigureEmbeddedDatabase.DatabaseProvider;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@AutoConfigureEmbeddedDatabase(provider = DatabaseProvider.ZONKY)
class ProjectAndRecipeIT {

    @Autowired
    MockMvc mvc;

    @Autowired
    ObjectMapper json;

    @Test
    void fullLifecycle_createProject_addRecipe_renderAndPersist_listAndFetchImage() throws Exception {
        // 1. Create project
        MvcResult projectRes = mvc.perform(post("/projects")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"name":"Sandbox","description":"e2e test","ownerId":"alice"}
                                """))
                .andDo(print())
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.name", is("Sandbox")))
                .andExpect(jsonPath("$.ownerId", is("alice")))
                .andExpect(jsonPath("$.recipeCount", is(0)))
                .andReturn();
        String projectId = json.readTree(projectRes.getResponse().getContentAsString())
                .get("id").asText();

        // 2. Add recipe
        String recipeBody = """
                {
                  "name":"classic mandel",
                  "recipe": {
                    "viewport": {"xMin":-2.0,"xMax":1.0,"yMin":-1.2,"yMax":1.2},
                    "renderSettings": {"widthPx":64,"heightPx":64},
                    "colorSettings": {"paletteName":"fire"},
                    "fractalType": "mandelbrot",
                    "params": {"maxIter":80,"escapeRadius":2.0,"smoothing":true}
                  }
                }
                """;
        MvcResult recipeRes = mvc.perform(post("/projects/" + projectId + "/recipes")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(recipeBody))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.name", is("classic mandel")))
                .andExpect(jsonPath("$.fractalType", is("mandelbrot")))
                .andExpect(jsonPath("$.recipe.params.maxIter", is(80)))
                .andReturn();
        String recipeId = json.readTree(recipeRes.getResponse().getContentAsString())
                .get("id").asText();

        // 3. recipeCount on project should now be 1
        mvc.perform(get("/projects/" + projectId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.recipeCount", is(1)));

        // 4. List recipes for project
        mvc.perform(get("/projects/" + projectId + "/recipes"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()", is(1)))
                .andExpect(jsonPath("$[0].id", is(recipeId)));

        // 5. Render-and-persist twice → two render rows
        MvcResult firstRender = mvc.perform(post("/recipes/" + recipeId + "/renders"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.recipeId", is(recipeId)))
                .andExpect(jsonPath("$.imageUrl", notNullValue()))
                .andExpect(jsonPath("$.fileSizeBytes", greaterThanOrEqualTo(0)))
                .andExpect(jsonPath("$.performance.totalMs", greaterThanOrEqualTo(0)))
                .andReturn();
        JsonNode renderJson = json.readTree(firstRender.getResponse().getContentAsString());
        String renderId = renderJson.get("id").asText();
        assertEquals("/renders/" + renderId + "/image", renderJson.get("imageUrl").asText());

        mvc.perform(post("/recipes/" + recipeId + "/renders?includeBase64=true"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.imageBase64", notNullValue()));

        mvc.perform(get("/recipes/" + recipeId + "/renders"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()", is(2)));

        // 6. Fetch the PNG bytes — must start with PNG magic
        MvcResult imageRes = mvc.perform(get("/renders/" + renderId + "/image"))
                .andExpect(status().isOk())
                .andReturn();
        byte[] png = imageRes.getResponse().getContentAsByteArray();
        assertTrue(png.length > 100, "PNG should be non-trivial in size, got " + png.length);
        assertEquals((byte) 0x89, png[0]);
        assertEquals('P', png[1]);
        assertEquals('N', png[2]);
        assertEquals('G', png[3]);
        assertEquals(MediaType.IMAGE_PNG_VALUE, imageRes.getResponse().getContentType());

        // 7. Delete project → cascade deletes recipes and renders
        mvc.perform(delete("/projects/" + projectId))
                .andExpect(status().isNoContent());
        mvc.perform(get("/projects/" + projectId))
                .andExpect(status().isNotFound());
        mvc.perform(get("/recipes/" + recipeId))
                .andExpect(status().isNotFound());
    }

    @Test
    void recipeOnMissingProjectReturns404() throws Exception {
        String body = """
                {"name":"x","recipe":{"viewport":{"xMin":-2.0,"xMax":1.0,"yMin":-1.2,"yMax":1.2},
                "renderSettings":{"widthPx":32,"heightPx":32},
                "colorSettings":{"paletteName":"fire"},
                "fractalType":"mandelbrot",
                "params":{"maxIter":50,"escapeRadius":2.0,"smoothing":true}}}
                """;
        mvc.perform(post("/projects/00000000-0000-0000-0000-000000000000/recipes")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error", is("not_found")));
    }

    @Test
    void renderAndPersistOnMissingRecipeReturns404() throws Exception {
        mvc.perform(post("/recipes/00000000-0000-0000-0000-000000000000/renders"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error", is("not_found")));
    }

    @Test
    void recipeJsonRoundTripsThroughJsonbColumn() throws Exception {
        // Create project + recipe with all four fractal families to exercise the
        // sealed-interface JSONB converter end-to-end.
        MvcResult projectRes = mvc.perform(post("/projects")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"name":"jsonb-roundtrip","ownerId":"bob"}
                                """))
                .andExpect(status().isCreated())
                .andReturn();
        String projectId = json.readTree(projectRes.getResponse().getContentAsString())
                .get("id").asText();

        String[][] cases = {
                {"mandelbrot", "{\"maxIter\":50,\"escapeRadius\":2.0,\"smoothing\":true}"},
                {"julia",      "{\"cRe\":-0.7,\"cIm\":0.27015,\"maxIter\":50,\"escapeRadius\":2.0,\"smoothing\":true}"},
                {"burning_ship","{\"maxIter\":50,\"escapeRadius\":2.0,\"smoothing\":false}"},
                {"multibrot",  "{\"exponent\":4,\"maxIter\":50,\"escapeRadius\":2.0,\"smoothing\":true}"},
        };
        for (String[] c : cases) {
            String type = c[0];
            String params = c[1];
            String body = """
                    {"name":"%s","recipe":{
                      "viewport":{"xMin":-1.5,"xMax":1.5,"yMin":-1.5,"yMax":1.5},
                      "renderSettings":{"widthPx":32,"heightPx":32},
                      "colorSettings":{"paletteName":"fire"},
                      "fractalType":"%s",
                      "params":%s
                    }}
                    """.formatted(type, type, params);
            MvcResult res = mvc.perform(post("/projects/" + projectId + "/recipes")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(body))
                    .andExpect(status().isCreated())
                    .andReturn();
            String recipeId = json.readTree(res.getResponse().getContentAsString())
                    .get("id").asText();

            // Fetch back — recipe.fractalType should reflect on read.
            MvcResult fetched = mvc.perform(get("/recipes/" + recipeId))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.fractalType", is(type)))
                    .andReturn();
            JsonNode reread = json.readTree(fetched.getResponse().getContentAsString());
            assertNotNull(reread.get("recipe").get("params"),
                    "params should round-trip for " + type);
        }
    }
}
