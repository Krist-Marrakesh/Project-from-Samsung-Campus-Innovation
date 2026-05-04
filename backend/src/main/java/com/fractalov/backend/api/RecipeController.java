package com.fractalov.backend.api;

import com.fractalov.backend.api.dto.RecipeRequests;
import com.fractalov.backend.api.dto.RecipeResponses;
import com.fractalov.backend.service.persistence.RecipeService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.ResponseStatus;

import java.util.List;
import java.util.UUID;

@RestController
public class RecipeController {

    private final RecipeService recipes;

    public RecipeController(RecipeService recipes) {
        this.recipes = recipes;
    }

    @PostMapping("/projects/{projectId}/recipes")
    @ResponseStatus(HttpStatus.CREATED)
    public RecipeResponses.View create(@PathVariable UUID projectId,
                                       @Valid @RequestBody RecipeRequests.Create req) {
        return RecipeResponses.View.of(recipes.create(projectId, req.name(), req.recipe()));
    }

    @GetMapping("/projects/{projectId}/recipes")
    public List<RecipeResponses.View> listByProject(@PathVariable UUID projectId) {
        return recipes.listByProject(projectId).stream()
                .map(RecipeResponses.View::of)
                .toList();
    }

    @GetMapping("/recipes/{id}")
    public RecipeResponses.View get(@PathVariable UUID id) {
        return RecipeResponses.View.of(recipes.get(id));
    }

    @DeleteMapping("/recipes/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@PathVariable UUID id) {
        recipes.delete(id);
    }
}
