package com.fractalov.backend.api;

import com.fractalov.backend.dto.FractalType;
import com.fractalov.backend.service.color.PaletteRegistry;
import com.fractalov.backend.service.render.RenderDispatcher;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
public class MetadataController {

    private final RenderDispatcher dispatcher;
    private final PaletteRegistry paletteRegistry;

    public MetadataController(RenderDispatcher dispatcher, PaletteRegistry paletteRegistry) {
        this.dispatcher = dispatcher;
        this.paletteRegistry = paletteRegistry;
    }

    @GetMapping("/fractal-types")
    public List<String> fractalTypes() {
        return dispatcher.supportedTypes().stream().map(FractalType::asJson).toList();
    }

    @GetMapping("/palettes")
    public List<String> palettes() {
        return paletteRegistry.names();
    }
}
