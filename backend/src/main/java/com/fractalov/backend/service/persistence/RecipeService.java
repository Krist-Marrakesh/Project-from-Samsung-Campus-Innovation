package com.fractalov.backend.service.persistence;

import com.fractalov.backend.domain.entity.RecipeEntity;
import com.fractalov.backend.domain.repo.ProjectRepository;
import com.fractalov.backend.domain.repo.RecipeRepository;
import com.fractalov.backend.dto.FractalRecipe;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;

@Service
public class RecipeService {

    private final RecipeRepository recipes;
    private final ProjectRepository projects;

    public RecipeService(RecipeRepository recipes, ProjectRepository projects) {
        this.recipes = recipes;
        this.projects = projects;
    }

    @Transactional
    public RecipeEntity create(UUID projectId, String name, FractalRecipe recipe) {
        if (!projects.existsById(projectId)) {
            throw new NotFoundException("project", projectId);
        }
        return recipes.save(RecipeEntity.newRecipe(projectId, name, recipe));
    }

    @Transactional(readOnly = true)
    public RecipeEntity get(UUID id) {
        return recipes.findById(id)
                .orElseThrow(() -> new NotFoundException("recipe", id));
    }

    @Transactional(readOnly = true)
    public List<RecipeEntity> listByProject(UUID projectId) {
        if (!projects.existsById(projectId)) {
            throw new NotFoundException("project", projectId);
        }
        return recipes.findByProjectId(projectId);
    }

    @Transactional
    public void delete(UUID id) {
        if (!recipes.existsById(id)) {
            throw new NotFoundException("recipe", id);
        }
        recipes.deleteById(id);
    }
}
