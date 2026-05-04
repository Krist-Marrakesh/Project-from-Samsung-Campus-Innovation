package com.fractalov.backend.domain.repo;

import com.fractalov.backend.domain.entity.RenderJobEntity;
import org.springframework.data.jdbc.repository.query.Modifying;
import org.springframework.data.jdbc.repository.query.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.data.repository.query.Param;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface RenderJobRepository extends CrudRepository<RenderJobEntity, UUID> {

    @Query("SELECT * FROM render_jobs WHERE recipe_id = :recipeId ORDER BY queued_at DESC")
    List<RenderJobEntity> findByRecipeId(UUID recipeId);

    /**
     * Atomic claim of the oldest queued job, tagged with a caller-supplied
     * {@code token}. Combines:
     * <ul>
     *   <li>{@code FOR UPDATE SKIP LOCKED} — multiple workers never grab the same row</li>
     *   <li>Tagging UPDATE with {@code claim_token} so the immediate re-fetch
     *       ({@link #findJustClaimed}) can find this exact row even if two pollers
     *       claim within the same Postgres millisecond.</li>
     * </ul>
     * Spring Data JDBC requires {@code @Modifying} for an UPDATE; the
     * {@code SELECT … FOR UPDATE} is wrapped in a CTE so a single statement does the
     * whole claim.
     */
    @Modifying
    @Query("""
            WITH next_job AS (
                SELECT id
                FROM render_jobs
                WHERE status = 'queued'
                ORDER BY queued_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE render_jobs j
               SET status = 'running',
                   started_at = :now,
                   claim_token = :token,
                   attempt_count = attempt_count + 1
              FROM next_job
             WHERE j.id = next_job.id
            """)
    int claimOne(@Param("now") Instant now, @Param("token") UUID token);

    /** Re-fetches the row just claimed by {@link #claimOne(Instant, UUID)}. The token
     * uniquely identifies this claim, so the lookup is collision-free even under
     * concurrent claims with identical {@code started_at}. */
    @Query("""
            SELECT * FROM render_jobs
             WHERE claim_token = :token
             LIMIT 1
            """)
    Optional<RenderJobEntity> findJustClaimed(@Param("token") UUID token);

    /**
     * Sweep that runs at boot to recover from a process kill: any job that survived
     * shutdown in queued/running state is fenced as failed.
     */
    @Modifying
    @Query("""
            UPDATE render_jobs
               SET status = 'failed',
                   error_message = :reason,
                   finished_at = :now,
                   claim_token = NULL
             WHERE status IN ('queued', 'running')
            """)
    int markStuckAsFailed(@Param("now") Instant now, @Param("reason") String reason);

    @Modifying
    @Query("""
            UPDATE render_jobs
               SET status = 'cancelled',
                   finished_at = :now
             WHERE id = :id AND status = 'queued'
            """)
    int cancelIfQueued(@Param("id") UUID id, @Param("now") Instant now);

    @Query("SELECT count(*) FROM render_jobs WHERE status = 'queued'")
    long countQueued();

    @Query("SELECT count(*) FROM render_jobs WHERE status = 'running'")
    long countRunning();
}
