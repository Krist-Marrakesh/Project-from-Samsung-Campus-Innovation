package com.fractalov.backend.domain.repo;

import com.fractalov.backend.domain.entity.ProjectEntity;
import org.springframework.data.jdbc.repository.query.Query;
import org.springframework.data.repository.CrudRepository;

import java.util.List;
import java.util.UUID;

public interface ProjectRepository extends CrudRepository<ProjectEntity, UUID> {

    @Query("SELECT * FROM projects ORDER BY created_at DESC")
    List<ProjectEntity> findAllOrderedByCreatedDesc();

    @Query("SELECT * FROM projects WHERE owner_id = :ownerId ORDER BY created_at DESC")
    List<ProjectEntity> findByOwnerId(String ownerId);
}
