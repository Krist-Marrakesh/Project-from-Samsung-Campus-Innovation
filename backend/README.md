# fractalov-backend

Standalone Spring Boot 3 / Java 21 server. Stage 4 adds asynchronous render jobs — Postgres-as-queue with `FOR UPDATE SKIP LOCKED`, a worker pool, and SSE progress streams. Earlier stages still apply: four fractal families with SSAA + histogram coloring (Stage 2), persistent projects → recipes → renders (Stage 3). No Redis, no Celery, no Docker — Postgres + a Spring `@Async` pool covers what we need without a second moving part.

The module is **not** wired into the Android Gradle build. Run it with its own wrapper:

```bash
cd backend
./gradlew build          # compile + run integration tests against an embedded Postgres
./gradlew bootRun        # start on :8080 against your local Postgres
```

## First-time setup (Postgres)

The dev profile expects a real Postgres at `localhost:5432`. With Homebrew Postgres 16:

```bash
brew install postgresql@16
brew services start postgresql@16
```

Create the database and role used by `application.yml` defaults:

```bash
psql -h localhost -d postgres <<'SQL'
CREATE ROLE fractalov WITH LOGIN PASSWORD 'fractalov_dev' CREATEDB;
CREATE DATABASE fractalov_dev OWNER fractalov;
SQL
```

Override via env vars if your setup differs:

```
FRACTALOV_DB_URL       jdbc:postgresql://localhost:5432/fractalov_dev
FRACTALOV_DB_USER      fractalov
FRACTALOV_DB_PASSWORD  fractalov_dev
FRACTALOV_RENDERS_ROOT ./data/renders
```

Flyway runs `V1__init.sql` on startup; you don't pre-create tables.

Tests use Zonky's embedded Postgres (binaries downloaded once into the Gradle cache, no Docker). Each `@SpringBootTest` class spins up an isolated DB.

Gradle 8.13 itself runs on JDK 17 (pinned via `org.gradle.java.home` in `gradle.properties`); the toolchain downloads JDK 21 automatically via the foojay resolver to compile the code. Change the path in `gradle.properties` if your JDK 17 install lives elsewhere.

## Endpoints

### Stateless compute (Stage 1–2)

| Method | Path               | Purpose                                            |
|--------|--------------------|----------------------------------------------------|
| GET    | `/health`          | Liveness check                                     |
| GET    | `/fractal-types`   | Supported fractal families                         |
| GET    | `/palettes`        | Registered colour palettes                         |
| POST   | `/validate-recipe` | Validate a recipe without rendering, HTTP 200      |
| POST   | `/render`          | Render the recipe and return base64 PNG + metadata |

### Persistence (Stage 3)

| Method | Path                                  | Purpose                                                  |
|--------|---------------------------------------|----------------------------------------------------------|
| POST   | `/projects`                           | Create a project                                         |
| GET    | `/projects`                           | List all projects (with `recipeCount`)                   |
| GET    | `/projects/{id}`                      | Get one project + recipe count                           |
| PUT    | `/projects/{id}`                      | Rename / re-describe a project                           |
| DELETE | `/projects/{id}`                      | Delete (cascades to recipes and renders)                 |
| POST   | `/projects/{pid}/recipes`             | Save a `FractalRecipe` under a project                   |
| GET    | `/projects/{pid}/recipes`             | List recipes for a project                               |
| GET    | `/recipes/{id}`                       | Get one saved recipe (full payload, ready to re-render)  |
| DELETE | `/recipes/{id}`                       | Delete a recipe (cascades to its renders)                |
| POST   | `/recipes/{rid}/renders`              | **Sync** render of a saved recipe; persists row + PNG    |
| GET    | `/recipes/{rid}/renders`              | List renders made from a recipe                          |
| GET    | `/renders/{id}`                       | Get render metadata + `imageUrl`                         |
| GET    | `/renders/{id}/image`                 | Download the rendered PNG                                |

`POST /recipes/{rid}/renders?includeBase64=true` also embeds the PNG in the JSON response — useful for mobile clients that want both the persisted record and an immediate preview without a second round-trip.

The original `POST /render` is unchanged: it still computes-and-returns base64 without saving anything. Persistence is opt-in via the saved-recipe endpoints.

### Async jobs (Stage 4)

| Method | Path                                    | Purpose                                                                  |
|--------|-----------------------------------------|--------------------------------------------------------------------------|
| POST   | `/recipes/{rid}/render-jobs`            | Enqueue a render job; HTTP 202 with `{id, status:"queued"}`              |
| GET    | `/render-jobs/{id}`                     | Poll job state — `queued/running/succeeded/failed/cancelled`             |
| GET    | `/render-jobs/{id}/stream`              | SSE: snapshot + live transitions until terminal                          |
| POST   | `/render-jobs/{id}/cancel`              | Cancel iff still queued; running jobs are not interrupted (by design)    |
| GET    | `/recipes/{rid}/render-jobs`            | List jobs for a recipe                                                   |

Successful jobs end with `renderId` and `imageUrl` populated — the artefact is the same `renders` row + on-disk PNG that the sync endpoint produces, so async vs sync is a transport choice, not a data-shape choice.

On validation failure for `/render`, the handler returns HTTP 400 with `ErrorResponse { requestId, error, message, details[] }`. `/validate-recipe` always returns HTTP 200 with `ValidationResponse { valid, errors[] }`.

## Fractal families

| `fractalType`   | Formula                                         | Extra params           |
|-----------------|-------------------------------------------------|------------------------|
| `mandelbrot`    | `z ↦ z² + c`, `z₀ = 0`                          | —                      |
| `julia`         | `z ↦ z² + c`, `c` fixed, `z₀ = (x, y)`          | `cRe`, `cIm`           |
| `burning_ship`  | `z ↦ (|Re z| + i|Im z|)² + c`, `z₀ = 0`         | —                      |
| `multibrot`     | `z ↦ zᴺ + c`, `z₀ = 0`, `N ∈ [2, 10]`           | `exponent`             |

All families share `maxIter`, `escapeRadius`, `smoothing`. `fractalType` lives only at the recipe level — Jackson uses it as an external discriminator to decode `params`, so sibling-field drift is impossible.

## Render envelope

Request:

```json
{
  "recipe": {
    "viewport": {"xMin": -2.0, "xMax": 1.0, "yMin": -1.2, "yMax": 1.2},
    "renderSettings": {"widthPx": 1024, "heightPx": 1024, "samplesPerAxis": 2},
    "colorSettings": {"paletteName": "fire", "mode": "histogram"},
    "fractalType": "mandelbrot",
    "params": {"maxIter": 500, "escapeRadius": 2.0, "smoothing": true}
  }
}
```

`samplesPerAxis` and `colorSettings.mode` are optional (defaults: `1` and `"linear"`).

Response (truncated):

```json
{
  "requestId": "8f36...",
  "status": "ok",
  "imageBase64": "iVBORw0KGgo...",
  "format": "png",
  "widthPx": 1024,
  "heightPx": 1024,
  "performance": {
    "renderMs": 30,
    "colorizeMs": 13,
    "encodeMs": 25,
    "totalMs": 68
  },
  "recipeEcho": { "...": "..." }
}
```

## Rendering features

### Supersampling (SSAA)

`renderSettings.samplesPerAxis ∈ [1, 3]` (default 1). Each output pixel averages `N × N` jittered sub-samples inside the pixel cell. If every sub-sample is in-set the output pixel stays in-set (sentinel `-1.0`); otherwise escape samples are arithmetic-meaned. Cost scales with `N²`.

### Colour modes

`colorSettings.mode`:

- `"linear"` (default): normalise escape value by `maxIter`, look up palette.
- `"histogram"`: rank all escape values by CDF percentile and look up palette. Removes banding in smooth gradients; most noticeable combined with `smoothing: true` because fractional escape values give a dense CDF.

### Architecture notes

- `FractalRenderer` is an interface; each family (`MandelbrotRenderer`, `JuliaRenderer`, `BurningShipRenderer`, `MultibrotRenderer`) is a Spring `@Component` that returns `RenderResult { double[][] escapeMap, int maxIter, long durationMs }`.
- Coordinate walking and SSAA averaging live in one place — `GridSweep` — and each renderer just supplies a stateless `PixelKernel` lambda (capturing params by value). This removed the duplicated parallel loop that would have appeared once 4 renderers existed.
- `RenderDispatcher` collects all `FractalRenderer` beans into `Map<FractalType, FractalRenderer>` once at startup.
- `Colorizer` is separate from the math so the raw escape-time field can later be consumed by the ML pipeline (Stage 5). Two modes live there now; adding a new one is a branch inside one method.
- Hot loops use `IntStream.range(0, height).parallel()` with flat `double zRe, zIm` state — no `Complex` boxing, JIT friendly. `MultibrotRenderer` falls back to polar form (atan2/pow) since `zᴺ` has no cheap closed form beyond N=2 — this is visible in the baseline numbers below.

## curl cheat-sheet

Mandelbrot:

```bash
curl -s -X POST localhost:8080/render \
  -H 'Content-Type: application/json' \
  -d '{"recipe":{"viewport":{"xMin":-2.0,"xMax":1.0,"yMin":-1.2,"yMax":1.2},"renderSettings":{"widthPx":512,"heightPx":512},"colorSettings":{"paletteName":"fire"},"fractalType":"mandelbrot","params":{"maxIter":200,"escapeRadius":2.0,"smoothing":true}}}' \
  | jq -r .imageBase64 | base64 -d > /tmp/mandel.png && open /tmp/mandel.png
```

Burning Ship:

```bash
curl -s -X POST localhost:8080/render \
  -H 'Content-Type: application/json' \
  -d '{"recipe":{"viewport":{"xMin":-2.0,"xMax":1.5,"yMin":-2.0,"yMax":1.0},"renderSettings":{"widthPx":512,"heightPx":512},"colorSettings":{"paletteName":"fire"},"fractalType":"burning_ship","params":{"maxIter":200,"escapeRadius":2.0,"smoothing":true}}}' \
  | jq -r .imageBase64 | base64 -d > /tmp/ship.png && open /tmp/ship.png
```

Multibrot N=5 with rainbow-cyclic palette:

```bash
curl -s -X POST localhost:8080/render \
  -H 'Content-Type: application/json' \
  -d '{"recipe":{"viewport":{"xMin":-1.5,"xMax":1.5,"yMin":-1.5,"yMax":1.5},"renderSettings":{"widthPx":512,"heightPx":512},"colorSettings":{"paletteName":"rainbow_cyclic","mode":"histogram"},"fractalType":"multibrot","params":{"exponent":5,"maxIter":200,"escapeRadius":2.0,"smoothing":true}}}' \
  | jq -r .imageBase64 | base64 -d > /tmp/multi5.png && open /tmp/multi5.png
```

Julia with SSAA:

```bash
curl -s -X POST localhost:8080/render \
  -H 'Content-Type: application/json' \
  -d '{"recipe":{"viewport":{"xMin":-1.5,"xMax":1.5,"yMin":-1.5,"yMax":1.5},"renderSettings":{"widthPx":512,"heightPx":512,"samplesPerAxis":2},"colorSettings":{"paletteName":"ocean"},"fractalType":"julia","params":{"cRe":-0.7,"cIm":0.27015,"maxIter":200,"escapeRadius":2.0,"smoothing":true}}}' \
  | jq -r .imageBase64 | base64 -d > /tmp/julia.png && open /tmp/julia.png
```

## Palettes

`grayscale`, `fire`, `ocean`, `rainbow_cyclic`.

In-set points (reached `maxIter` without escape) are always rendered as pure black. Escape-time values are mapped to `[0, 1]` by the selected colour mode before palette lookup.

## Baseline performance

Measurements on Apple Silicon (darwin/arm64), JDK 21 toolchain, parallelStream over rows. Each row is the full `/render` wall-clock.

```
1024 × 1024, mandelbrot, maxIter=500, smoothing=true, palette=fire, linear
    run=1 renderMs=29  colorizeMs=27  encodeMs=24  totalMs=82
    run=2 renderMs=30  colorizeMs=9   encodeMs=24  totalMs=65
    run=3 renderMs=30  colorizeMs=18  encodeMs=25  totalMs=74
    run=4 renderMs=29  colorizeMs=13  encodeMs=25  totalMs=68

1024 × 1024, mandelbrot, SSAA=2 (4 samples/pixel)
    run=1 renderMs=127 colorizeMs=8   encodeMs=25  totalMs=161
    run=2 renderMs=119 colorizeMs=9   encodeMs=26  totalMs=154
    → ~4× render time, matches 4× sample budget.

1024 × 1024, multibrot N=3, maxIter=500
    run=1 renderMs=844 colorizeMs=11  encodeMs=23  totalMs=879
    run=2 renderMs=813 colorizeMs=7   encodeMs=25  totalMs=846
    → ~27× slower than mandelbrot on same maxIter (atan2 + pow per iter).
      Stage 10 candidate: replace polar form with direct z^N expansion for small N.
```

Re-measure whenever a renderer's hot loop or the `GridSweep` walker changes.

## Schema

Three tables, one Flyway migration (`db/migration/V1__init.sql`):

```
projects (id UUID PK, name, description, owner_id default 'anonymous',
          created_at, updated_at)
  └── recipes (id UUID PK, project_id FK CASCADE, name, fractal_type,
               recipe_json JSONB, version, created_at)
        └── renders (id UUID PK, recipe_id FK CASCADE, image_path,
                     width_px, height_px, palette_name, color_mode,
                     samples_per_axis, render_ms, colorize_ms, encode_ms,
                     total_ms, file_size_bytes, created_at)
```

`recipe_json` is the full serialised `FractalRecipe` (same shape as the wire format, same external `fractalType` discriminator). The custom `FractalRecipeJsonConverters` shares the web `ObjectMapper`, so on-disk and on-wire shapes never drift.

`owner_id` is a stub for Stage 7+ auth; it defaults to `'anonymous'` and the API accepts (but does not authenticate) an explicit value.

## Filesystem layout for renders

```
{app.storage.renders-root}/
  └── 2026-04-25/                         # date the render was made
      ├── 6b614808-...-c68d.png
      └── 2abd3280-...-12b2.png
```

Only the relative path (`2026-04-25/{render_id}.png`) lives in the DB, so the storage tier can be relocated by changing `app.storage.renders-root`. Stage 8 will swap the filesystem for S3-style object storage behind the same `ImageStorage` interface.

## Persistence flow walkthrough

```bash
PROJECT_ID=$(curl -s -X POST localhost:8080/projects \
  -H 'Content-Type: application/json' \
  -d '{"name":"Sandbox","description":"e2e","ownerId":"krist"}' | jq -r .id)

RECIPE_ID=$(curl -s -X POST localhost:8080/projects/$PROJECT_ID/recipes \
  -H 'Content-Type: application/json' \
  -d '{"name":"classic mandel","recipe":{
        "viewport":{"xMin":-2.0,"xMax":1.0,"yMin":-1.2,"yMax":1.2},
        "renderSettings":{"widthPx":256,"heightPx":256},
        "colorSettings":{"paletteName":"fire","mode":"histogram"},
        "fractalType":"mandelbrot",
        "params":{"maxIter":300,"escapeRadius":2.0,"smoothing":true}
       }}' | jq -r .id)

# Render-and-persist; the PNG goes to disk, metadata to Postgres.
RENDER_ID=$(curl -s -X POST localhost:8080/recipes/$RECIPE_ID/renders | jq -r .id)

# Pull the saved PNG.
curl -s -o /tmp/saved.png localhost:8080/renders/$RENDER_ID/image
open /tmp/saved.png

# History.
curl -s localhost:8080/recipes/$RECIPE_ID/renders | jq '.[] | {id, performance, fileSizeBytes}'
```

## Architecture notes (Stage 3 additions)

- **Spring Data JDBC, not JPA.** Entities are records (`ProjectEntity`, `RecipeEntity`, `RenderEntity`); each implements `Persistable<UUID>` with `isNew() == createdAt == null` so the immutable record + UUID id + INSERT semantics all work together. JPA would have forced mutable getters and a session/lazy-init dance that doesn't fit the project's record-everywhere style.
- **JSONB via shared ObjectMapper.** `FractalRecipeJsonConverters` registers a `WritingConverter` and `ReadingConverter` that wrap the same Jackson instance the web layer uses. The sealed `FractalParams` hierarchy round-trips through Postgres exactly as it does over HTTP — verified by `recipeJsonRoundTripsThroughJsonbColumn` test.
- **Compute outside transactions.** `RenderHistoryService.renderAndPersist` runs the full render, colorize, encode, and file write *outside* any `@Transactional` boundary; only the row insert (handled by `CrudRepository.save`) is transactional. A 1-second render no longer holds a JDBC connection.
- **Image bytes never enter the DB.** `image_path` is a relative path; the only "blob-ish" thing in Postgres is the recipe JSONB (~1 KB).

## Async jobs (Stage 4)

### Why DB-as-queue, not Redis or RabbitMQ

We already pay the Postgres tax. A second broker would cost a service, a port, a healthcheck, and a Docker entry — and buy us nothing for a single backend instance. Postgres' `FOR UPDATE SKIP LOCKED` on an indexed status column is a textbook queue, gives us transactional integrity ("create job + render row in one TX"), and survives restart for free.

If/when we go multi-instance or need cross-service messaging, the swap is mechanical: `RenderJobRepository` is a small surface, and the wire API (submit / poll / SSE) doesn't care what's behind it.

### Wire flow

```bash
# 1. Enqueue
JOB_ID=$(curl -s -X POST localhost:8080/recipes/$RECIPE_ID/render-jobs | jq -r .id)

# 2a. Poll
curl -s localhost:8080/render-jobs/$JOB_ID
# {"status":"queued",  ...}
# {"status":"running", ...}
# {"status":"succeeded","renderId":"...","imageUrl":"/renders/.../image", ...}

# 2b. Or stream SSE — one connection, three events for a healthy run
curl -sN -H 'Accept: text/event-stream' localhost:8080/render-jobs/$JOB_ID/stream
# event:snapshot   data:{...status:"queued"...}
# event:running    data:{...status:"running"...}
# event:succeeded  data:{...renderId:"..."...}

# 3. Cancel (only effective while queued)
curl -X POST localhost:8080/render-jobs/$JOB_ID/cancel
```

### Internals

- **`render_jobs` table** (V2 migration). State machine: `queued → running → succeeded|failed`, plus `queued → cancelled`. Status check constraint enforces the alphabet at the DB level.
- **`RenderJobRepository.claimOne(now)`** runs `UPDATE … FROM (SELECT id … WHERE status='queued' ORDER BY queued_at FOR UPDATE SKIP LOCKED LIMIT 1)` in one statement. Two parallel pollers can never grab the same row; verified by `claimOneIsExclusive` test.
- **`RenderJobPoller`** (`@Scheduled fixedDelay=500ms`) keeps the executor's free slots filled. We track in-flight jobs in an `AtomicInteger` against `workerPoolSize` — if all workers are busy, we don't pull, so a queued row stays queued and is fair-shared if a second instance ever shows up.
- **`RenderJobWorker`** does the actual `RenderHistoryService.renderAndPersist` call, then transitions the row and publishes a `JobLifecycleEvent` on the in-process `JobEventBus`.
- **`JobEventBus` + SSE**. The controller subscribes per-emitter, sends a current-state `snapshot` event first, then forwards bus events until terminal — at which point the emitter completes. Subscribers register *after* the snapshot send, so the snapshot/event ordering is monotonic for the client.
- **`RestartRecoveryRunner`** at boot fences any `queued`/`running` rows from a previous process as `failed("server restart")`. Simplest correct recovery; a multi-instance setup would replace this with a heartbeat lease.

### Tunables (env vars)

| Variable                     | Default | Effect                                  |
|------------------------------|---------|-----------------------------------------|
| `FRACTALOV_JOBS_WORKERS`     | `1`     | Async pool size. Each render already uses parallelStream over CPU cores, so >1 only helps if you want pipelining (one render computing while another encodes PNG). |
| `FRACTALOV_JOBS_POLL_MS`     | `500`   | Poller fixed-delay. Lower = lower job pickup latency, higher DB chatter. |
| `FRACTALOV_JOBS_SSE_HEARTBEAT` | `15`  | Heartbeat seconds for SSE keepalive (planned; current SSE has no idle timeout). |

### What jobs do **not** do (yet)

- **No cancellation of running jobs.** The math kernel doesn't poll an interrupt flag, and adding one adds branches to the hot loop. We add this only if a real user needs it.
- **No retries.** A failed job stays failed; the client decides whether to resubmit. Retry policy is a future Stage 4.5 concern (exponential backoff + dead-letter).
- **No multi-instance fairness.** Single backend. The SKIP-LOCKED claim is multi-instance-safe by construction, but the in-process `JobEventBus` is not — clients hitting instance B for a job running on instance A would only get the `snapshot` event, not live updates. Stage 8+ swaps the bus for Postgres `LISTEN/NOTIFY` if needed.

## Out of scope for Stage 4

No S3, no Docker, no auth, no CORS, no ML. Those arrive in later stages.
