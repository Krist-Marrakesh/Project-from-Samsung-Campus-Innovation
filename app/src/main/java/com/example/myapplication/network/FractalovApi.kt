package com.example.myapplication.network

import android.util.Base64
import com.example.myapplication.domain.FractalRecipe
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.defaultRequest
import io.ktor.client.plugins.logging.LogLevel
import io.ktor.client.plugins.logging.Logging
import io.ktor.client.request.forms.MultiPartFormDataContent
import io.ktor.client.request.forms.formData
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsBytes
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.Headers
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import io.ktor.http.takeFrom
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json

/**
 * Single source of truth for talking to the Java backend.
 *
 * Why Ktor + OkHttp engine: Ktor is the native-Kotlin HTTP client (suspending,
 * structured-concurrency-friendly), and the OkHttp engine is the right fit on
 * Android — battle-tested timeouts/connection pooling and avoids Ktor's
 * default CIO engine which has some surprises with cleartext HTTP on
 * pre-API-28 devices.
 *
 * Cleartext HTTP traffic to `10.0.2.2:8080` (the AVD loopback alias) is
 * permitted by `network_security_config.xml`; release builds will need a
 * scheme bump or a separate config when we ship past dev.
 */
class FractalovApi(baseUrl: String) {

    @OptIn(kotlinx.serialization.ExperimentalSerializationApi::class)
    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
        // The discriminator is implied by the polymorphic field's
        // @JsonClassDiscriminator; this just forces classic discriminator-on-object
        // semantics rather than the (default) array form when needed.
        classDiscriminatorMode = kotlinx.serialization.json.ClassDiscriminatorMode.POLYMORPHIC
    }

    private val client = HttpClient(OkHttp) {
        install(ContentNegotiation) { json(json) }
        install(HttpTimeout) {
            requestTimeoutMillis = 60_000
            connectTimeoutMillis = 10_000
            socketTimeoutMillis = 60_000
        }
        install(Logging) {
            level = LogLevel.INFO
        }
        defaultRequest {
            url.takeFrom(baseUrl)
        }
    }

    /**
     * Stateless render via Stage 1's `/render` endpoint.
     * Returns the decoded PNG bytes ready to feed into Coil.
     */
    suspend fun render(recipe: FractalRecipe): RenderResult {
        val resp = client.post("/render") {
            contentType(ContentType.Application.Json)
            setBody(RenderRequestDto(recipe))
        }
        if (!resp.status.isSuccess()) {
            throw BackendException(resp.status, resp.bodyAsText().take(500))
        }
        val dto: RenderResponseDto = resp.body()
        val pngBytes = Base64.decode(dto.imageBase64, Base64.NO_WRAP)
        return RenderResult(
            pngBytes = pngBytes,
            widthPx = dto.widthPx,
            heightPx = dto.heightPx,
            renderMs = dto.performance.renderMs,
            totalMs = dto.performance.totalMs,
        )
    }

    /**
     * ML suggest+render+persist pipeline.
     *
     * In addition to the rendered PNG, returns the original image bytes
     * unchanged so the client can show "you uploaded X / model rendered Y"
     * side-by-side without re-reading the gallery URI later.
     */
    suspend fun mlRenderFromImage(imageBytes: ByteArray, filename: String): MlRenderResult {
        val resp = client.post("/ml/render-from-image") {
            setBody(MultiPartFormDataContent(formData {
                append(
                    key = "file",
                    value = imageBytes,
                    headers = Headers.build {
                        append(HttpHeaders.ContentType, "image/png")
                        append(HttpHeaders.ContentDisposition, "filename=\"$filename\"")
                    },
                )
            }))
        }
        if (!resp.status.isSuccess()) {
            throw BackendException(resp.status, resp.bodyAsText().take(500))
        }
        val dto: MlRenderResponseDto = resp.body()
        // Download the persisted PNG by following the imageUrl returned
        // in the response. It is a relative path served by the same backend.
        val img = client.get(dto.render.imageUrl)
        if (!img.status.isSuccess()) {
            throw BackendException(img.status, "image fetch failed: ${img.bodyAsText().take(200)}")
        }
        return MlRenderResult(
            family = dto.suggestion.family,
            familyConfidence = dto.suggestion.familyConfidence,
            cRe = dto.suggestion.cRe,
            cIm = dto.suggestion.cIm,
            renderId = dto.render.id,
            pngBytes = img.bodyAsBytes(),
            originalBytes = imageBytes,
            renderMs = dto.render.performance.renderMs,
            totalMs = dto.render.performance.totalMs,
            recipe = dto.suggestion.suggestedRecipe,
        )
    }

    /**
     * Variations endpoint — asks the ML side for ``count`` perturbed
     * recipes around the supplied one. The Python service returns
     * recipes only; rendering them is the client's job (we fan out parallel
     * calls in the variations ViewModel).
     */
    suspend fun variations(
        recipe: FractalRecipe,
        count: Int = 8,
        seed: Int = 0,
    ): List<FractalRecipe> {
        val resp = client.post("/ml/variations") {
            contentType(ContentType.Application.Json)
            setBody(MlVariationsRequestDto(recipe = recipe, count = count, seed = seed))
        }
        if (!resp.status.isSuccess()) {
            throw BackendException(resp.status, resp.bodyAsText().take(500))
        }
        val dto: MlVariationsResponseDto = resp.body()
        return dto.recipes
    }

    /** Shared Json instance — exposed so callers can serialise / deserialise
     *  recipes consistently with the wire shape (used by route arguments). */
    fun json(): Json = json

    fun close() = client.close()
}

private fun HttpStatusCode.isSuccess() = value in 200..299

class BackendException(val status: HttpStatusCode, val payload: String) :
    RuntimeException("backend HTTP ${status.value}: ${payload.take(200)}")

data class RenderResult(
    val pngBytes: ByteArray,
    val widthPx: Int,
    val heightPx: Int,
    val renderMs: Long,
    val totalMs: Long,
)

data class MlRenderResult(
    val family: String,
    val familyConfidence: Double,
    val cRe: Double?,
    val cIm: Double?,
    val renderId: String,
    val pngBytes: ByteArray,
    val originalBytes: ByteArray,
    val renderMs: Long,
    val totalMs: Long,
    val recipe: FractalRecipe,
)
