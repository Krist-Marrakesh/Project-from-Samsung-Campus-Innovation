package com.fractalov.backend.domain.repo;

import com.fractalov.backend.domain.entity.RecipeEntity;
import org.springframework.data.jdbc.repository.query.Query;
import org.springframework.data.repository.CrudRepository;

import java.util.List;
import java.util.UUID;

public interface RecipeRepository extends CrudRepository<RecipeEntity, UUID> {

    @Query("SELECT * FROM recipes WHERE project_id = :projectId ORDER BY created_at DESC")
    List<RecipeEntity> findByProjectId(UUID projectId);

    @Query("SELECT count(*) FROM recipes WHERE project_id = :projectId")
    long countByProjectId(UUID projectId);
}
