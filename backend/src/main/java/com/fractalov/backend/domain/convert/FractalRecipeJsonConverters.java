package com.fractalov.backend.domain.convert;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fractalov.backend.dto.FractalRecipe;
import org.postgresql.util.PGobject;
import org.springframework.data.convert.ReadingConverter;
import org.springframework.data.convert.WritingConverter;

import java.sql.SQLException;

/**
 * Bridges {@link FractalRecipe} to Postgres JSONB. Spring Data JDBC delegates value
 * conversion to the standard converter API, but knows nothing about Jackson — these
 * adapters wrap the same {@link ObjectMapper} used by the web layer so the on-disk
 * shape matches the wire shape exactly (same discriminator, same field names).
 */
public final class FractalRecipeJsonConverters {

    private FractalRecipeJsonConverters() {}

    @WritingConverter
    public static class Writing implements org.springframework.core.convert.converter.Converter<FractalRecipe, PGobject> {
        private final ObjectMapper mapper;

        public Writing(ObjectMapper mapper) {
            this.mapper = mapper;
        }

        @Override
        public PGobject convert(FractalRecipe source) {
            try {
                PGobject pg = new PGobject();
                pg.setType("jsonb");
                pg.setValue(mapper.writeValueAsString(source));
                return pg;
            } catch (JsonProcessingException | SQLException e) {
                throw new IllegalStateException("Failed to encode FractalRecipe as JSONB", e);
            }
        }
    }

    @ReadingConverter
    public static class Reading implements org.springframework.core.convert.converter.Converter<PGobject, FractalRecipe> {
        private final ObjectMapper mapper;

        public Reading(ObjectMapper mapper) {
            this.mapper = mapper;
        }

        @Override
        public FractalRecipe convert(PGobject source) {
            try {
                return mapper.readValue(source.getValue(), FractalRecipe.class);
            } catch (JsonProcessingException e) {
                throw new IllegalStateException("Failed to decode FractalRecipe from JSONB", e);
            }
        }
    }
}
