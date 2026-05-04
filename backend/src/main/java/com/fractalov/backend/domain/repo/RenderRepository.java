package com.fractalov.backend.domain.repo;

import com.fractalov.backend.domain.entity.RenderEntity;
import org.springframework.data.jdbc.repository.query.Query;
import org.springframework.data.repository.CrudRepository;

import java.util.List;
import java.util.UUID;

public interface RenderRepository extends CrudRepository<RenderEntity, UUID> {

    @Query("SELECT * FROM renders WHERE recipe_id = :recipeId ORDER BY created_at DESC")
    List<RenderEntity> findByRecipeId(UUID recipeId);
}
