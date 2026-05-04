package com.fractalov.backend.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fractalov.backend.domain.convert.FractalRecipeJsonConverters;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.jdbc.repository.config.AbstractJdbcConfiguration;

import java.util.List;

@Configuration
public class JdbcConfig extends AbstractJdbcConfiguration {

    private final ObjectMapper objectMapper;

    public JdbcConfig(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @Override
    protected List<?> userConverters() {
        return List.of(
                new FractalRecipeJsonConverters.Writing(objectMapper),
                new FractalRecipeJsonConverters.Reading(objectMapper)
        );
    }
}
