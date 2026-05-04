package com.fractalov.backend.dto.validation;

import com.fractalov.backend.dto.Viewport;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import jakarta.validation.ValidatorFactory;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ViewportValidatorTest {

    private static ValidatorFactory factory;
    private static Validator validator;

    @BeforeAll
    static void setup() {
        factory = Validation.buildDefaultValidatorFactory();
        validator = factory.getValidator();
    }

    @AfterAll
    static void tearDown() {
        factory.close();
    }

    @Test
    void acceptsNormalViewport() {
        Viewport v = new Viewport(-2.0, 1.0, -1.2, 1.2);
        assertTrue(validator.validate(v).isEmpty());
    }

    @Test
    void rejectsInvertedX() {
        Viewport v = new Viewport(1.0, -1.0, 0.0, 1.0);
        assertFalse(validator.validate(v).isEmpty());
    }

    @Test
    void rejectsInvertedY() {
        Viewport v = new Viewport(-1.0, 1.0, 1.0, -1.0);
        assertFalse(validator.validate(v).isEmpty());
    }

    @Test
    void rejectsDegenerateSpan() {
        Viewport v = new Viewport(0.5, 0.5, -1.0, 1.0);
        assertFalse(validator.validate(v).isEmpty());
    }

    @Test
    void rejectsNaN() {
        Viewport v = new Viewport(Double.NaN, 1.0, -1.0, 1.0);
        assertFalse(validator.validate(v).isEmpty());
    }
}
