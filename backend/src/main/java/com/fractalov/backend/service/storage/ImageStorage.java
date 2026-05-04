package com.fractalov.backend.service.storage;

import com.fractalov.backend.config.StorageProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.UUID;

/**
 * Persists rendered PNGs to a configurable filesystem root. Layout:
 * <pre>
 *   {root}/{yyyy-MM-dd}/{renderId}.png
 * </pre>
 * Returns a relative path (relative to the root) suitable for storing in the
 * {@code renders.image_path} column. Reading goes back through the same root —
 * absolute paths are never persisted in the DB so the storage tier can be
 * relocated by changing one config property.
 */
@Component
public class ImageStorage {

    private static final Logger log = LoggerFactory.getLogger(ImageStorage.class);
    private static final DateTimeFormatter DAY = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    public record StoredImage(String relativePath, long sizeBytes) {}

    private final Path root;

    public ImageStorage(StorageProperties props) {
        this.root = Paths.get(props.rendersRoot()).toAbsolutePath().normalize();
        try {
            Files.createDirectories(root);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot create renders root: " + root, e);
        }
        log.info("ImageStorage root: {}", root);
    }

    public StoredImage store(UUID renderId, BufferedImage image) {
        String day = LocalDate.now().format(DAY);
        String rel = day + "/" + renderId + ".png";
        Path target = root.resolve(rel);
        try {
            Files.createDirectories(target.getParent());
            if (!ImageIO.write(image, "png", target.toFile())) {
                throw new IllegalStateException("No PNG ImageWriter found in classpath");
            }
            long size = Files.size(target);
            return new StoredImage(rel, size);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to write PNG to " + target, e);
        }
    }

    public byte[] read(String relativePath) {
        Path target = resolveSafe(relativePath);
        try {
            return Files.readAllBytes(target);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read PNG at " + target, e);
        }
    }

    public boolean exists(String relativePath) {
        return Files.isRegularFile(resolveSafe(relativePath));
    }

    private Path resolveSafe(String relativePath) {
        Path candidate = root.resolve(relativePath).normalize();
        if (!candidate.startsWith(root)) {
            throw new IllegalArgumentException("Path escapes storage root: " + relativePath);
        }
        return candidate;
    }
}
