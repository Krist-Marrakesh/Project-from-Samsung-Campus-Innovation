package com.fractalov.backend.api;

import com.fractalov.backend.api.dto.ProjectRequests;
import com.fractalov.backend.api.dto.ProjectResponses;
import com.fractalov.backend.domain.entity.ProjectEntity;
import com.fractalov.backend.service.persistence.ProjectService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/projects")
public class ProjectController {

    private final ProjectService projects;

    public ProjectController(ProjectService projects) {
        this.projects = projects;
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public ProjectResponses.View create(@Valid @RequestBody ProjectRequests.Create req) {
        ProjectEntity saved = projects.create(req.name(), req.description(), req.ownerId());
        return ProjectResponses.View.of(saved, 0);
    }

    @GetMapping
    public List<ProjectResponses.View> list() {
        return projects.list().stream()
                .map(p -> ProjectResponses.View.of(p, projects.countRecipes(p.id())))
                .toList();
    }

    @GetMapping("/{id}")
    public ProjectResponses.View get(@PathVariable UUID id) {
        ProjectEntity p = projects.get(id);
        return ProjectResponses.View.of(p, projects.countRecipes(id));
    }

    @PutMapping("/{id}")
    public ProjectResponses.View update(@PathVariable UUID id, @Valid @RequestBody ProjectRequests.Update req) {
        ProjectEntity updated = projects.update(id, req.name(), req.description());
        return ProjectResponses.View.of(updated, projects.countRecipes(id));
    }

    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@PathVariable UUID id) {
        projects.delete(id);
    }
}
