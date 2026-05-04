package com.fractalov.backend.api;

import com.fractalov.backend.api.dto.MlRequests;
import com.fractalov.backend.api.dto.MlResponses;
import com.fractalov.backend.api.dto.RenderRecordResponses;
import com.fractalov.backend.service.ml.MlSuggestion;
import com.fractalov.backend.service.ml.MlSuggestionService;
import com.fractalov.backend.service.persistence.RenderHistoryService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.UncheckedIOException;

@RestController
@RequestMapping("/ml")
public class MlController {

    private final MlSuggestionService suggestions;

    public MlController(MlSuggestionService suggestions) {
        this.suggestions = suggestions;
    }

    @PostMapping(path = "/suggest-from-image", consumes = "multipart/form-data")
    public MlResponses.SuggestionView suggest(@RequestParam("file") MultipartFile file) {
        MlSuggestion s = suggestions.suggest(readBytes(file), file.getOriginalFilename());
        return MlResponses.SuggestionView.of(s);
    }

    @PostMapping(path = "/render-from-image", consumes = "multipart/form-data")
    public MlResponses.RenderFromImageView renderFromImage(@RequestParam("file") MultipartFile file) {
        MlSuggestionService.PersistedSuggestion ps =
                suggestions.suggestRenderAndPersist(readBytes(file), file.getOriginalFilename());
        RenderHistoryService.PersistedRender persisted = ps.persisted();
        RenderRecordResponses.View renderView = RenderRecordResponses.View.of(
                persisted.entity(),
                new com.fractalov.backend.dto.PerformanceBreakdown(
                        persisted.entity().renderMs(),
                        persisted.entity().colorizeMs(),
                        persisted.entity().encodeMs(),
                        persisted.entity().totalMs()
                ),
                null
        );
        return new MlResponses.RenderFromImageView(
                MlResponses.SuggestionView.of(ps.suggestion()),
                ps.recipe().projectId(),
                ps.recipe().id(),
                renderView
        );
    }

    @PostMapping("/variations")
    public MlResponses.VariationsView variations(@Valid @RequestBody MlRequests.VariationsRequest req) {
        return new MlResponses.VariationsView(
                suggestions.variations(req.recipe(), req.count(), req.seed())
        );
    }

    private byte[] readBytes(MultipartFile file) {
        try {
            return file.getBytes();
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read uploaded file", e);
        }
    }
}
