package com.fractalov.backend.service.color;

import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;

@Component
public class PaletteRegistry {

    private final Map<String, Palette> byName;

    public PaletteRegistry(List<Palette> palettes) {
        this.byName = palettes.stream()
                .collect(Collectors.toUnmodifiableMap(Palette::name, Function.identity()));
    }

    public Optional<Palette> find(String name) {
        if (name == null) return Optional.empty();
        return Optional.ofNullable(byName.get(name));
    }

    public Palette require(String name) {
        return find(name).orElseThrow(() ->
                new IllegalArgumentException("Unknown palette: " + name));
    }

    public List<String> names() {
        return byName.keySet().stream().sorted().toList();
    }
}
