package com.fractalov.backend.service.ml;

import com.fractalov.backend.domain.entity.ProjectEntity;
import com.fractalov.backend.domain.entity.RecipeEntity;
import com.fractalov.backend.domain.repo.ProjectRepository;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.service.persistence.ProjectService;
import com.fractalov.backend.service.persistence.RecipeService;
import com.fractalov.backend.service.persistence.RenderHistoryService;
import com.fractalov.backend.service.persistence.RenderHistoryService.PersistedRender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.List;
import java.util.UUID;

/**
 * Bridges the {@link MlClient} (Stage 7 inference HTTP) to the persistence
 * stack reused from Stage 3. The service offers two flavours:
 *
 * <ul>
 *   <li>"suggest only" — call inference, return the suggestion verbatim. No DB
 *       writes.</li>
 *   <li>"render and persist" — call inference, save the suggested recipe under
 *       a {@code "ML suggestions YYYY-MM-DD"} bucket project, run the existing
 *       persisted-render flow, return the resulting render row.</li>
 * </ul>
 *
 * The bucket project is auto-created on first use of the day; subsequent
 * suggestions on the same date append to it. This keeps the {@code projects}
 * table from being polluted by per-suggestion rows while still giving every
 * persisted render a parent project for cascade deletes.
 */
@Service
public class MlSuggestionService {

    private static final Logger log = LoggerFactory.getLogger(MlSuggestionService.class);
    private static final String SUGGESTION_OWNER = "ml";

    private final MlClient ml;
    private final ProjectRepository projects;
    private final ProjectService projectService;
    private final RecipeService recipeService;
    private final RenderHistoryService renderHistory;

    public MlSuggestionService(
            MlClient ml,
            ProjectRepository projects,
            ProjectService projectService,
            RecipeService recipeService,
            RenderHistoryService renderHistory) {
        this.ml = ml;
        this.projects = projects;
        this.projectService = projectService;
        this.recipeService = recipeService;
        this.renderHistory = renderHistory;
    }

    public MlSuggestion suggest(byte[] image, String filename) {
        return ml.suggestFromImage(image, filename);
    }

    public PersistedSuggestion suggestRenderAndPersist(byte[] image, String filename) {
        MlSuggestion suggestion = ml.suggestFromImage(image, filename);
        ProjectEntity bucket = bucketProject();
        RecipeEntity recipeRow = recipeService.create(
                bucket.id(),
                "ml-" + suggestion.family() + "-" + UUID.randomUUID().toString().substring(0, 8),
                suggestion.recipe()
        );
        PersistedRender persisted = renderHistory.renderAndPersist(recipeRow.id(), false);
        log.info("ml render-from-image projectId={} recipeId={} renderId={} family={} conf={}",
                bucket.id(), recipeRow.id(), persisted.entity().id(),
                suggestion.family(), suggestion.familyConfidence());
        return new PersistedSuggestion(suggestion, recipeRow, persisted);
    }

    public List<FractalRecipe> variations(FractalRecipe recipe, int count, int seed) {
        return ml.variations(recipe, count, seed);
    }

    private ProjectEntity bucketProject() {
        // One project per UTC day for the synthetic owner. Linear scan over
        // the (small) ML project list is cheap and avoids a custom query.
        String name = "ML suggestions " + LocalDate.now();
        return projects.findByOwnerId(SUGGESTION_OWNER).stream()
                .filter(p -> name.equals(p.name()))
                .findFirst()
                .orElseGet(() -> projectService.create(
                        name,
                        "auto-created bucket for ML suggestions",
                        SUGGESTION_OWNER
                ));
    }

    public record PersistedSuggestion(
            MlSuggestion suggestion,
            RecipeEntity recipe,
            PersistedRender persisted
    ) {}
}
