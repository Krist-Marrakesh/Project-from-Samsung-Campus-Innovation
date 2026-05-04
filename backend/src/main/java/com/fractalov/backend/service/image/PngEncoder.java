package com.fractalov.backend.service.image;

import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Base64;

@Component
public class PngEncoder {

    public String encodeToBase64(BufferedImage image) {
        ByteArrayOutputStream buf = new ByteArrayOutputStream(64 * 1024);
        try {
            if (!ImageIO.write(image, "png", buf)) {
                throw new IllegalStateException("No PNG ImageWriter found in classpath");
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to encode PNG", e);
        }
        return Base64.getEncoder().encodeToString(buf.toByteArray());
    }
}
