package com.fractalov.backend.service.persistence;

import com.fractalov.backend.domain.entity.RecipeEntity;
import com.fractalov.backend.domain.entity.RenderEntity;
import com.fractalov.backend.domain.repo.RecipeRepository;
import com.fractalov.backend.domain.repo.RenderRepository;
import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.PerformanceBreakdown;
import com.fractalov.backend.service.color.Colorizer;
import com.fractalov.backend.service.color.Palette;
import com.fractalov.backend.service.color.PaletteRegistry;
import com.fractalov.backend.service.image.PngEncoder;
import com.fractalov.backend.service.render.RenderDispatcher;
import com.fractalov.backend.service.render.RenderResult;
import com.fractalov.backend.service.storage.ImageStorage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.UUID;

/**
 * Renders a saved {@link FractalRecipe} and persists three artefacts:
 * the on-disk PNG, the {@code renders} row, and a summary structure used by
 * the API layer. Compute happens outside the JDBC transaction (long-running),
 * the transactional boundary covers only the row insert.
 */
@Service
public class RenderHistoryService {

    private static final Logger log = LoggerFactory.getLogger(RenderHistoryService.class);

    private final RecipeRepository recipes;
    private final RenderRepository renders;
    private final RenderDispatcher dispatcher;
    private final Colorizer colorizer;
    private final PngEncoder pngEncoder;
    private final PaletteRegistry palettes;
    private final ImageStorage storage;

    public RenderHistoryService(
            RecipeRepository recipes,
            RenderRepository renders,
            RenderDispatcher dispatcher,
            Colorizer colorizer,
            PngEncoder pngEncoder,
            PaletteRegistry palettes,
            ImageStorage storage) {
        this.recipes = recipes;
        this.renders = renders;
        this.dispatcher = dispatcher;
        this.colorizer = colorizer;
        this.pngEncoder = pngEncoder;
        this.palettes = palettes;
        this.storage = storage;
    }

    public record PersistedRender(RenderEntity entity, String imageBase64) {}

    public PersistedRender renderAndPersist(UUID recipeId, boolean includeBase64) {
        RecipeEntity recipeRow = recipes.findById(recipeId)
                .orElseThrow(() -> new NotFoundException("recipe", recipeId));
        FractalRecipe recipe = recipeRow.recipeJson();
        Palette palette = palettes.require(recipe.colorSettings().paletteName());

        long totalStart = System.nanoTime();
        RenderResult result = dispatcher.dispatch(recipe);

        long colorStart = System.nanoTime();
        BufferedImage image = colorizer.colorize(
                result,
                recipe.viewport(),
                palette,
                recipe.colorSettings().effectiveMode()
        );
        long colorizeMs = (System.nanoTime() - colorStart) / 1_000_000L;

        long encodeStart = System.nanoTime();
        String base64 = includeBase64 ? pngEncoder.encodeToBase64(image) : null;
        long encodeMs = (System.nanoTime() - encodeStart) / 1_000_000L;

        // Pre-allocate the render id so the on-disk PNG path can use it, then INSERT
        // a row carrying that id. The row is treated as new because RenderEntity
        // implements Persistable and reports isNew() based on createdAt being null,
        // not on the id being null — so we keep an immutable record + UUID id +
        // INSERT semantics.
        UUID renderId = UUID.randomUUID();
        ImageStorage.StoredImage stored = storage.store(renderId, image);

        long totalMs = (System.nanoTime() - totalStart) / 1_000_000L;

        RenderEntity row = new RenderEntity(
                renderId,
                recipeId,
                stored.relativePath(),
                recipe.renderSettings().widthPx(),
                recipe.renderSettings().heightPx(),
                recipe.colorSettings().paletteName(),
                recipe.colorSettings().effectiveMode().name().toLowerCase(),
                (short) recipe.renderSettings().effectiveSamplesPerAxis(),
                result.durationMs(),
                colorizeMs,
                encodeMs,
                totalMs,
                stored.sizeBytes(),
                null
        );
        RenderEntity saved = renders.save(row);

        log.info("rendered+persisted id={} recipe={} {}x{} render={}ms colorize={}ms encode={}ms total={}ms file={}b path={}",
                saved.id(), recipeId,
                row.widthPx(), row.heightPx(),
                row.renderMs(), row.colorizeMs(), row.encodeMs(), row.totalMs(),
                stored.sizeBytes(), stored.relativePath());

        return new PersistedRender(saved, base64);
    }

    @Transactional(readOnly = true)
    public RenderEntity get(UUID renderId) {
        return renders.findById(renderId)
                .orElseThrow(() -> new NotFoundException("render", renderId));
    }

    @Transactional(readOnly = true)
    public List<RenderEntity> listByRecipe(UUID recipeId) {
        if (!recipes.existsById(recipeId)) {
            throw new NotFoundException("recipe", recipeId);
        }
        return renders.findByRecipeId(recipeId);
    }

    public PerformanceBreakdown perfOf(RenderEntity row) {
        return new PerformanceBreakdown(row.renderMs(), row.colorizeMs(), row.encodeMs(), row.totalMs());
    }

    public byte[] readImage(RenderEntity row) {
        return storage.read(row.imagePath());
    }
}
