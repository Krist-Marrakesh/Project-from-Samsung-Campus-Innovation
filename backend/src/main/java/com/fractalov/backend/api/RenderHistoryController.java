package com.fractalov.backend.api;

import com.fractalov.backend.api.dto.RenderRecordResponses;
import com.fractalov.backend.domain.entity.RenderEntity;
import com.fractalov.backend.service.persistence.RenderHistoryService;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.UUID;

@RestController
public class RenderHistoryController {

    private final RenderHistoryService history;

    public RenderHistoryController(RenderHistoryService history) {
        this.history = history;
    }

    @PostMapping("/recipes/{recipeId}/renders")
    public RenderRecordResponses.View renderAndPersist(
            @PathVariable UUID recipeId,
            @RequestParam(name = "includeBase64", defaultValue = "false") boolean includeBase64) {
        RenderHistoryService.PersistedRender result = history.renderAndPersist(recipeId, includeBase64);
        return RenderRecordResponses.View.of(
                result.entity(),
                history.perfOf(result.entity()),
                result.imageBase64()
        );
    }

    @GetMapping("/recipes/{recipeId}/renders")
    public List<RenderRecordResponses.View> listByRecipe(@PathVariable UUID recipeId) {
        return history.listByRecipe(recipeId).stream()
                .map(r -> RenderRecordResponses.View.of(r, history.perfOf(r), null))
                .toList();
    }

    @GetMapping("/renders/{id}")
    public RenderRecordResponses.View get(@PathVariable UUID id) {
        RenderEntity row = history.get(id);
        return RenderRecordResponses.View.of(row, history.perfOf(row), null);
    }

    @GetMapping("/renders/{id}/image")
    public ResponseEntity<byte[]> image(@PathVariable UUID id) {
        RenderEntity row = history.get(id);
        byte[] bytes = history.readImage(row);
        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_PNG)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "inline; filename=\"" + row.id() + ".png\"")
                .body(bytes);
    }
}
