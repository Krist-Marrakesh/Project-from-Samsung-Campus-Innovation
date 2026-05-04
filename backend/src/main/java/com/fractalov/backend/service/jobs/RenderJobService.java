package com.fractalov.backend.service.jobs;

import com.fractalov.backend.domain.entity.RenderJobEntity;
import com.fractalov.backend.domain.repo.RecipeRepository;
import com.fractalov.backend.domain.repo.RenderJobRepository;
import com.fractalov.backend.service.persistence.NotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

@Service
public class RenderJobService {

    private final RenderJobRepository jobs;
    private final RecipeRepository recipes;

    public RenderJobService(RenderJobRepository jobs, RecipeRepository recipes) {
        this.jobs = jobs;
        this.recipes = recipes;
    }

    @Transactional
    public RenderJobEntity submit(UUID recipeId) {
        if (!recipes.existsById(recipeId)) {
            throw new NotFoundException("recipe", recipeId);
        }
        return jobs.save(RenderJobEntity.newQueued(recipeId));
    }

    @Transactional(readOnly = true)
    public RenderJobEntity get(UUID id) {
        return jobs.findById(id).orElseThrow(() -> new NotFoundException("render_job", id));
    }

    @Transactional(readOnly = true)
    public List<RenderJobEntity> listByRecipe(UUID recipeId) {
        if (!recipes.existsById(recipeId)) {
            throw new NotFoundException("recipe", recipeId);
        }
        return jobs.findByRecipeId(recipeId);
    }

    /**
     * Cancels a job iff it is still queued. A running job is not cancellable in
     * this stage — proper cooperative cancellation needs the math kernel to poll
     * an interrupt flag, which we'd add only if a user actually needs it.
     */
    @Transactional
    public boolean cancel(UUID id) {
        if (!jobs.existsById(id)) {
            throw new NotFoundException("render_job", id);
        }
        return jobs.cancelIfQueued(id, Instant.now()) > 0;
    }
}
