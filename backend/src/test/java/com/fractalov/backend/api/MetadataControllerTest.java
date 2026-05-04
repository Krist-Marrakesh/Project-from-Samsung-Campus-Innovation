package com.fractalov.backend.api;

import io.zonky.test.db.AutoConfigureEmbeddedDatabase;
import io.zonky.test.db.AutoConfigureEmbeddedDatabase.DatabaseProvider;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.hasItems;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@AutoConfigureEmbeddedDatabase(provider = DatabaseProvider.ZONKY)
class MetadataControllerTest {

    @Autowired
    MockMvc mvc;

    @Test
    void listsFractalTypes() throws Exception {
        mvc.perform(get("/fractal-types"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", containsInAnyOrder(
                        "mandelbrot", "julia", "burning_ship", "multibrot")));
    }

    @Test
    void listsPalettes() throws Exception {
        mvc.perform(get("/palettes"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasItems("grayscale", "fire", "ocean", "rainbow_cyclic")));
    }
}
