package com.fractalov.backend.api;

import com.fractalov.backend.dto.FractalRecipe;
import com.fractalov.backend.dto.PerformanceBreakdown;
import com.fractalov.backend.dto.RenderRequest;
import com.fractalov.backend.dto.RenderResponse;
import com.fractalov.backend.dto.ValidationResponse;
import com.fractalov.backend.service.color.Colorizer;
import com.fractalov.backend.service.color.Palette;
import com.fractalov.backend.service.color.PaletteRegistry;
import com.fractalov.backend.service.image.PngEncoder;
import com.fractalov.backend.service.render.RenderDispatcher;
import com.fractalov.backend.service.render.RenderResult;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.validation.BeanPropertyBindingResult;
import org.springframework.validation.FieldError;
import org.springframework.validation.SmartValidator;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.UUID;

@RestController
public class RenderController {

    private static final Logger log = LoggerFactory.getLogger(RenderController.class);

    private final RenderDispatcher renderDispatcher;
    private final PaletteRegistry paletteRegistry;
    private final Colorizer colorizer;
    private final PngEncoder pngEncoder;
    private final SmartValidator validator;

    public RenderController(RenderDispatcher renderDispatcher,
                            PaletteRegistry paletteRegistry,
                            Colorizer colorizer,
                            PngEncoder pngEncoder,
                            SmartValidator validator) {
        this.renderDispatcher = renderDispatcher;
        this.paletteRegistry = paletteRegistry;
        this.colorizer = colorizer;
        this.pngEncoder = pngEncoder;
        this.validator = validator;
    }

    @PostMapping("/validate-recipe")
    public ValidationResponse validate(@RequestBody RenderRequest request) {
        BeanPropertyBindingResult errors = new BeanPropertyBindingResult(request, "renderRequest");
        validator.validate(request, errors);

        List<ValidationResponse.FieldViolation> violations = errors.getFieldErrors().stream()
                .map(this::toViolation)
                .toList();

        if (violations.isEmpty() && request.recipe() != null) {
            String paletteName = request.recipe().colorSettings() == null ? null
                    : request.recipe().colorSettings().paletteName();
            if (paletteName != null && paletteRegistry.find(paletteName).isEmpty()) {
                violations = List.of(new ValidationResponse.FieldViolation(
                        "recipe.colorSettings.paletteName",
                        "unknown palette: " + paletteName));
            }
        }

        return new ValidationResponse(violations.isEmpty(), violations);
    }

    @PostMapping("/render")
    public RenderResponse render(@Valid @RequestBody RenderRequest request) {
        FractalRecipe recipe = request.recipe();
        Palette palette = paletteRegistry.require(recipe.colorSettings().paletteName());

        long totalStart = System.nanoTime();
        RenderResult result = renderDispatcher.dispatch(recipe);

        long colorStart = System.nanoTime();
        BufferedImage image = colorizer.colorize(
                result,
                recipe.viewport(),
                palette,
                recipe.colorSettings().effectiveMode()
        );
        long colorizeMs = (System.nanoTime() - colorStart) / 1_000_000L;

        long encodeStart = System.nanoTime();
        String base64 = pngEncoder.encodeToBase64(image);
        long encodeMs = (System.nanoTime() - encodeStart) / 1_000_000L;

        long totalMs = (System.nanoTime() - totalStart) / 1_000_000L;

        String requestId = UUID.randomUUID().toString();
        log.info("render id={} type={} {}x{} ssaa={} mode={} render={}ms colorize={}ms encode={}ms total={}ms",
                requestId,
                recipe.fractalType(),
                recipe.renderSettings().widthPx(),
                recipe.renderSettings().heightPx(),
                recipe.renderSettings().effectiveSamplesPerAxis(),
                recipe.colorSettings().effectiveMode(),
                result.durationMs(),
                colorizeMs,
                encodeMs,
                totalMs);

        return new RenderResponse(
                requestId,
                "ok",
                base64,
                "png",
                recipe.renderSettings().widthPx(),
                recipe.renderSettings().heightPx(),
                new PerformanceBreakdown(result.durationMs(), colorizeMs, encodeMs, totalMs),
                recipe
        );
    }

    private ValidationResponse.FieldViolation toViolation(FieldError e) {
        String field = e.getField().isEmpty() ? e.getObjectName() : e.getField();
        return new ValidationResponse.FieldViolation(field, e.getDefaultMessage());
    }
}
