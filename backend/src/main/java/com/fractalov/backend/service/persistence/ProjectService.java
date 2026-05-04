package com.fractalov.backend.service.persistence;

import com.fractalov.backend.domain.entity.ProjectEntity;
import com.fractalov.backend.domain.repo.ProjectRepository;
import com.fractalov.backend.domain.repo.RecipeRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;

@Service
public class ProjectService {

    private final ProjectRepository projects;
    private final RecipeRepository recipes;

    public ProjectService(ProjectRepository projects, RecipeRepository recipes) {
        this.projects = projects;
        this.recipes = recipes;
    }

    @Transactional
    public ProjectEntity create(String name, String description, String ownerId) {
        return projects.save(ProjectEntity.newProject(name, description, ownerId));
    }

    @Transactional(readOnly = true)
    public List<ProjectEntity> list() {
        return projects.findAllOrderedByCreatedDesc();
    }

    @Transactional(readOnly = true)
    public ProjectEntity get(UUID id) {
        return projects.findById(id)
                .orElseThrow(() -> new NotFoundException("project", id));
    }

    @Transactional(readOnly = true)
    public long countRecipes(UUID projectId) {
        return recipes.countByProjectId(projectId);
    }

    @Transactional
    public ProjectEntity update(UUID id, String name, String description) {
        ProjectEntity existing = get(id);
        return projects.save(existing.withUpdated(name, description));
    }

    @Transactional
    public void delete(UUID id) {
        if (!projects.existsById(id)) {
            throw new NotFoundException("project", id);
        }
        projects.deleteById(id);
    }
}
