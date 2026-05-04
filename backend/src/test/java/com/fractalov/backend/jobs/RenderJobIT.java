package com.fractalov.backend.jobs;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fractalov.backend.domain.entity.JobStatus;
import com.fractalov.backend.domain.entity.RenderJobEntity;
import com.fractalov.backend.domain.repo.RenderJobRepository;
import com.fractalov.backend.service.jobs.RenderJobWorker;
import io.zonky.test.db.AutoConfigureEmbeddedDatabase;
import io.zonky.test.db.AutoConfigureEmbeddedDatabase.DatabaseProvider;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.time.Duration;
import java.time.Instant;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static org.awaitility.Awaitility.await;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@AutoConfigureEmbeddedDatabase(provider = DatabaseProvider.ZONKY)
class RenderJobIT {

    @Autowired
    MockMvc mvc;

    @Autowired
    ObjectMapper json;

    @Autowired
    RenderJobRepository jobs;

    @Autowired
    RenderJobWorker worker;

    @Test
    void submitJobThenPollerDrivesItToSucceeded() throws Exception {
        UUID recipeId = createProjectAndRecipe("mandelbrot",
                "{\"maxIter\":40,\"escapeRadius\":2.0,\"smoothing\":true}");

        MvcResult res = mvc.perform(post("/recipes/" + recipeId + "/render-jobs"))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.status", is("queued")))
                .andExpect(jsonPath("$.id", notNullValue()))
                .andReturn();
        UUID jobId = UUID.fromString(json.readTree(res.getResponse().getContentAsString())
                .get("id").asText());

        // Poller runs every 500ms in real config; give it plenty of slack.
        await().atMost(20, TimeUnit.SECONDS)
                .pollInterval(Duration.ofMillis(200))
                .untilAsserted(() -> {
                    RenderJobEntity row = jobs.findById(jobId).orElseThrow();
                    assertEquals(JobStatus.SUCCEEDED, row.status(),
                            "job did not reach SUCCEEDED, last state=" + row.status()
                                    + " err=" + row.errorMessage());
                });

        // The job's render_id must point at a real renders row, with downloadable PNG.
        mvc.perform(get("/render-jobs/" + jobId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status", is("succeeded")))
                .andExpect(jsonPath("$.renderId", notNullValue()))
                .andExpect(jsonPath("$.imageUrl", notNullValue()));

        RenderJobEntity finalRow = jobs.findById(jobId).orElseThrow();
        MvcResult png = mvc.perform(get("/renders/" + finalRow.renderId() + "/image"))
                .andExpect(status().isOk())
                .andReturn();
        byte[] bytes = png.getResponse().getContentAsByteArray();
        assertTrue(bytes.length > 100);
        assertEquals((byte) 0x89, bytes[0]);
        assertEquals('P', bytes[1]);
    }

    @Test
    void cancelOnQueuedJobMarksItCancelled() throws Exception {
        UUID recipeId = createProjectAndRecipe("mandelbrot",
                "{\"maxIter\":40,\"escapeRadius\":2.0,\"smoothing\":false}");

        // Insert a queued job directly (avoid race with poller picking it up first).
        RenderJobEntity queued = jobs.save(RenderJobEntity.newQueued(recipeId));

        mvc.perform(post("/render-jobs/" + queued.id() + "/cancel"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status", is("cancelled")));

        RenderJobEntity after = jobs.findById(queued.id()).orElseThrow();
        assertEquals(JobStatus.CANCELLED, after.status());
        assertNotNull(after.finishedAt());
    }

    @Test
    void claimOneIsExclusive() {
        // Two parallel claims on a single queued row must produce only one transition.
        UUID recipeId = createRecipeOrFail();
        RenderJobEntity row = jobs.save(RenderJobEntity.newQueued(recipeId));

        Instant now = Instant.now();
        UUID t1 = UUID.randomUUID();
        UUID t2 = UUID.randomUUID();
        int firstClaim = jobs.claimOne(now, t1);
        int secondClaim = jobs.claimOne(now, t2);

        assertEquals(1, firstClaim, "first claim should grab the row");
        assertEquals(0, secondClaim, "second claim must find no queued row");
        assertEquals(JobStatus.RUNNING, jobs.findById(row.id()).orElseThrow().status());

        // findJustClaimed must locate the row exclusively by the winning token, even
        // though the losing token never wrote to any row.
        assertTrue(jobs.findJustClaimed(t1).isPresent(),
                "winning token must locate its row");
        assertTrue(jobs.findJustClaimed(t2).isEmpty(),
                "losing token must locate no row");
    }

    @Test
    void renderJobOnMissingRecipeReturns404() throws Exception {
        mvc.perform(post("/recipes/00000000-0000-0000-0000-000000000000/render-jobs"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error", is("not_found")));
    }

    private UUID createProjectAndRecipe(String type, String paramsJson) throws Exception {
        MvcResult p = mvc.perform(post("/projects")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"jobs-it\",\"ownerId\":\"jobs-it\"}"))
                .andExpect(status().isCreated())
                .andReturn();
        String projectId = json.readTree(p.getResponse().getContentAsString()).get("id").asText();

        String body = """
                {"name":"r","recipe":{
                  "viewport":{"xMin":-2.0,"xMax":1.0,"yMin":-1.2,"yMax":1.2},
                  "renderSettings":{"widthPx":48,"heightPx":48},
                  "colorSettings":{"paletteName":"fire"},
                  "fractalType":"%s",
                  "params":%s
                }}
                """.formatted(type, paramsJson);
        MvcResult r = mvc.perform(post("/projects/" + projectId + "/recipes")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isCreated())
                .andReturn();
        return UUID.fromString(json.readTree(r.getResponse().getContentAsString())
                .get("id").asText());
    }

    private UUID createRecipeOrFail() {
        try {
            return createProjectAndRecipe("mandelbrot",
                    "{\"maxIter\":40,\"escapeRadius\":2.0,\"smoothing\":false}");
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}
