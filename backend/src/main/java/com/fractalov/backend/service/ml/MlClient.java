package com.fractalov.backend.service.ml;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fractalov.backend.config.MlProperties;
import com.fractalov.backend.dto.FractalRecipe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublisher;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Calls the Python {@code fractalov-ml serve} inference HTTP server.
 *
 * <p>Why JDK {@link HttpClient} and a hand-written multipart body instead of
 * Spring {@code RestClient}: the Python server uses the standard FastAPI
 * multipart parser which is strict about the {@code Content-Disposition}
 * header — it requires {@code name="file"} on the part. Spring's
 * {@code FormHttpMessageConverter} sets that header correctly, but the
 * conversion path through {@link org.springframework.web.client.RestClient}
 * with a {@code MultiValueMap} body has been ignoring the part-level
 * {@code HttpHeaders} we set on the {@code ByteArrayResource}, producing a
 * 422 from FastAPI. Hand-rolling the body is two short methods, removes the
 * surprise, and stays trivially debuggable from a {@code curl} comparison.
 */
@Component
public class MlClient {

    private static final Logger log = LoggerFactory.getLogger(MlClient.class);
    private static final String BOUNDARY_PREFIX = "----fractalov";

    private final MlProperties props;
    private final HttpClient client;
    private final ObjectMapper json;

    public MlClient(MlProperties props, ObjectMapper json) {
        this.props = props;
        this.json = json;
        // Pin HTTP/1.1. Default JDK HttpClient negotiates HTTP/2 via the
        // {@code Upgrade: h2c} header on cleartext connections — uvicorn
        // (and FastAPI's multipart parser) reads the {@code Upgrade} header
        // as a protocol switch hint and discards the request body, which
        // surfaces as a 422 "Field required: file". Sticking to HTTP/1.1
        // makes the wire trivially diff-able against curl.
        this.client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofMillis(Math.min(props.timeoutMs(), 5_000)))
                .build();
    }

    public boolean enabled() {
        return props.enabled();
    }

    private void requireEnabled() {
        if (!props.enabled()) {
            throw new MlUnavailableException(
                    "ml is disabled (set app.ml.enabled=true and start fractalov-ml serve)");
        }
    }

    public MlSuggestion suggestFromImage(byte[] imageBytes, String filename) {
        requireEnabled();
        String boundary = BOUNDARY_PREFIX + UUID.randomUUID().toString().replace("-", "");
        byte[] body = buildMultipart(boundary, "file",
                filename == null ? "input.png" : filename, "image/png", imageBytes);

        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(props.url()).resolve("/infer/suggest-from-image"))
                .timeout(Duration.ofMillis(props.timeoutMs()))
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(BodyPublishers.ofByteArray(body))
                .build();

        HttpResponse<byte[]> resp = sendOrThrow(req, "suggest-from-image");
        if (resp.statusCode() / 100 != 2) {
            String text = new String(resp.body(), StandardCharsets.UTF_8);
            throw new MlUnavailableException("ml suggest-from-image returned HTTP "
                    + resp.statusCode() + ": " + truncate(text, 500));
        }
        SuggestionDto dto = readJson(resp.body(), SuggestionDto.class, "suggest-from-image");
        return new MlSuggestion(
                dto.family,
                dto.familyConfidence,
                dto.familyDistribution == null ? Map.of() : dto.familyDistribution,
                dto.cRe,
                dto.cIm,
                dto.recipe
        );
    }

    public List<FractalRecipe> variations(FractalRecipe recipe, int count, int seed) {
        requireEnabled();
        VariationsRequestDto reqBody = new VariationsRequestDto(recipe, count, seed);
        byte[] payload;
        try {
            payload = json.writeValueAsBytes(reqBody);
        } catch (IOException e) {
            throw new MlUnavailableException("could not serialise variations request", e);
        }

        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(props.url()).resolve("/infer/variations"))
                .timeout(Duration.ofMillis(props.timeoutMs()))
                .header("Content-Type", "application/json")
                .POST(BodyPublishers.ofByteArray(payload))
                .build();

        HttpResponse<byte[]> resp = sendOrThrow(req, "variations");
        if (resp.statusCode() / 100 != 2) {
            throw new MlUnavailableException("ml variations returned HTTP " + resp.statusCode()
                    + ": " + truncate(new String(resp.body(), StandardCharsets.UTF_8), 500));
        }
        VariationsResponseDto dto = readJson(resp.body(), VariationsResponseDto.class, "variations");
        return dto.recipes == null ? List.of() : dto.recipes;
    }

    private HttpResponse<byte[]> sendOrThrow(HttpRequest req, String op) {
        try {
            return client.send(req, HttpResponse.BodyHandlers.ofByteArray());
        } catch (Exception ex) {
            log.warn("ml {} transport failure: {}", op, ex.getMessage());
            throw new MlUnavailableException("ml " + op + " transport failure: " + ex.getMessage(), ex);
        }
    }

    private <T> T readJson(byte[] payload, Class<T> type, String op) {
        try {
            return json.readValue(payload, type);
        } catch (IOException e) {
            throw new MlUnavailableException("ml " + op + " body deserialisation failed", e);
        }
    }

    /**
     * Build a single-part multipart/form-data body. Header layout matches the
     * FastAPI {@code UploadFile} expectation: a part named {@code file} with a
     * {@code filename=} attribute and explicit {@code Content-Type}.
     */
    private static byte[] buildMultipart(
            String boundary, String name, String filename, String contentType, byte[] data) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(data.length + 256);
        try {
            String header = "--" + boundary + "\r\n"
                    + "Content-Disposition: form-data; name=\"" + name + "\"; filename=\"" + filename + "\"\r\n"
                    + "Content-Type: " + contentType + "\r\n\r\n";
            out.write(header.getBytes(StandardCharsets.UTF_8));
            out.write(data);
            out.write(("\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new IllegalStateException("multipart assembly failed (in-memory)", e);
        }
        return out.toByteArray();
    }

    private static String truncate(String s, int max) {
        return s.length() <= max ? s : s.substring(0, max) + "...";
    }

    public Duration timeout() {
        return Duration.ofMillis(props.timeoutMs());
    }

    /** Body of the FastAPI ``SuggestionResponse``. */
    public record SuggestionDto(
            String family,
            @JsonProperty("family_confidence") double familyConfidence,
            @JsonProperty("family_distribution") Map<String, Double> familyDistribution,
            @JsonProperty("c_re") Double cRe,
            @JsonProperty("c_im") Double cIm,
            FractalRecipe recipe
    ) {}

    public record VariationsRequestDto(FractalRecipe recipe, int count, int seed) {}

    public record VariationsResponseDto(List<FractalRecipe> recipes) {}
}
