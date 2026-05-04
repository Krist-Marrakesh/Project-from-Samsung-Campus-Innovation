# fractalov

Three-tier fractal exploration system for Android, built as a research-portfolio project. The user opens the app, picks a fractal family, the rendering happens server-side on a Java backend, and an ML head can be invoked to suggest a recipe from a sample image.

```
┌──────────────────────┐    HTTP/JSON     ┌──────────────────────┐    HTTP/JSON     ┌──────────────────────┐
│  Android client      │  ──────────────► │  Java backend        │  ──────────────► │  Python ML service   │
│  Kotlin + Compose    │  ◄── PNG / JSON  │  Spring Boot 3       │  ◄── Suggestion  │  FastAPI + PyTorch   │
│  app/                │                  │  backend/            │                  │  ml/                 │
└──────────────────────┘                  │     │ JDBC           │                  └──────────────────────┘
                                          │     ▼                │
                                          │  Postgres + Flyway   │
                                          └──────────────────────┘
```

Each tier is a **standalone project** with its own toolchain, lockfile, and README — they only ever meet over HTTP.

## Layout

```
.
├── app/                     Android client (Kotlin 2.0, Jetpack Compose, Ktor)
├── backend/                 Java 21 Spring Boot 3 server (rendering + persistence + ML proxy)
├── ml/                      Python 3.13 ML pipeline (dataset + training + FastAPI inference)
├── build.gradle.kts         Top-level Gradle (app/ only)
├── settings.gradle.kts      include(":app") — backend/ and ml/ are NOT submodules
├── gradle/libs.versions.toml
└── README.md                you are here
```

## Research thesis

Earlier stages built the three-tier system; later slices reframed it as one
cohesive research artifact:

* **Analytical generation as the source of truth** — a generalised escape-time
  engine (one parallel walker, four iteration kernels) renders any recipe
  deterministically.
* **ML as inverse inference, not decoration** — given an image, predict the
  generator parameters, with calibrated uncertainty and an optional optimisation
  refinement step.
* **Systematic study of the trade-offs** — performance broken down by stage,
  reconstruction quality measured against ground truth, calibration measured
  per split (in-distribution / near-OOD / hard-overlap).

The four families (Mandelbrot, Julia, Burning Ship, Multibrot) live behind a
single `IterationKernel` interface; downstream code is family-agnostic. The
synthetic dataset is treated as a frozen artifact via a YAML config + a
manifest of SHA-256 checksums.

## Stages, in build order

Earlier stages built the three-tier system; **slices 0–4** restructure it
around the research thesis above.

| Stage | Where it lives | What was added | README link |
|---|---|---|---|
| **1** | `backend/src/main/java/com/fractalov/backend/{api,dto,service}` | Stateless `/render` endpoint, Mandelbrot + Julia, base64 PNG over JSON, validation | [backend/README.md](backend/README.md) |
| **2** | `backend/.../service/render/{GridSweep,BurningShipRenderer,MultibrotRenderer}` + `service/color/Colorizer` | BurningShip + Multibrot, supersampling, histogram coloring, per-stage perf breakdown | same |
| **3** | `backend/.../domain` + `db/migration/V1__init.sql` + `service/persistence` + `service/storage` | Postgres + Flyway, projects → recipes → renders, PNGs on the filesystem | same |
| **4** | `backend/.../service/jobs` + `db/migration/V2__render_jobs.sql` | Async render jobs with `FOR UPDATE SKIP LOCKED`, SSE progress streams | same |
| **5** | `ml/src/fractalov_ml/{recipes,client,generate,splits,dataset,inspect}` | Dataset generation against the live backend, parquet labels, PyTorch `Dataset` | [ml/README.md](ml/README.md) |
| **6** | `ml/.../{models,training}` | Two-headed FractalCNN (family + Julia c), Optuna tuning, eval, reconstruction | same |
| **7** | `ml/.../{inference,serve}` + `backend/.../service/ml` + `MlController` | FastAPI inference server, Java `MlClient`, `/ml/*` endpoints in the backend | same |
| **8** | `app/src/main/java/com/example/myapplication/**` | Compose UI, Ktor client, two screens (preset render, ML-from-image) | this README, below |
| **Slice 0** | `backend/.../service/render/{EscapeTimeEngine,IterationKernel,FieldsOut,FieldStack,kernel/*}` + `service/color/Colorizer` + `db/migration/V3__claim_token.sql` | Generalised escape-time engine; four kernels behind one interface; distance-estimation channel (Mandelbrot/Julia/Multibrot); cardioid + period-2 short-circuits in the production Mandelbrot kernel; `DISTANCE_ESTIMATE` colour mode; `Colorizer` rewrite using `DataBufferInt` + quantile LUT; `claim_token` UUID column to make the job poller multi-instance-safe; `/bench/*` gated by `app.bench.enabled` | [backend/README.md](backend/README.md) |
| **Slice 1** | `backend/.../service/render/{StageStats,StageBreakdown,BenchmarkPipeline,BenchmarkScenario,BenchmarkPresets,MatrixBenchmarkRunner,MatrixResult,MatrixEntry}` + `api/BenchmarkController` + `backend/scripts/run-matrix.sh` | Benchmark harness as a research artifact: per-stage timing breakdown (`render` / `colorize` / `encode` / `total`), curated research preset matrix (~13 scenarios sweeping every axis), full Cartesian preset, `/bench/{scenarios, matrix, matrix-from-presets}` endpoints, CSV-pivot driver script | same |
| **Slice 2** | `ml/src/fractalov_ml/{dataset_config,recipes_v2,build_dataset,manifest}.py` + `ml/configs/dataset/research-v1.yaml` | Frozen synthetic dataset: YAML config, per-split RNG seed via SHA-256 of split name (orthogonal streams), parameter range overrides per split, five canonical splits (`train`, `val`, `test`, `near_ood`, `hard_overlap`), SHA-256 manifest of every PNG + the parquet, single `labels.parquet` with embedded `split` column | [ml/README.md](ml/README.md) |
| **Slice 3** | `ml/src/fractalov_ml/{dataset_v2,torch_renderer}.py` + `ml/.../models/cnn_v2.py` + `ml/.../training/{config_v2,loss_v2,train_v2,eval_v2,refine}.py` | Inverse-inference pipeline v2: multi-task with calibrated uncertainty (MVE Gaussian-NLL heads for Julia c and Multibrot exponent), differentiable PyTorch renderer (Mandelbrot/Julia/Multibrot), hybrid NN-warm-start + L-BFGS reconstruction refinement, per-split eval (test/near_ood/hard_overlap) including calibration σ-coverage and pre→post reconstruction MSE | same |
| **Slice 4a** | `ml/.../training/tune_v2.py` | Optuna tuning under the v2 search space: lr, weight_decay, batch_size, `c_weight`, `exp_weight`, `vp_weight`, `base_channels`, augmentation. Objective is val cls_loss — same metric `train-v2` uses for `best.pt` selection. `train-v2 --from-best` reads the produced `best_params_v2.json` | same |
| **Slice 4b** | `ml/.../torch_renderer.py` (Burning Ship + tensor-viewport grid) + `ml/.../models/cnn_v2.py` (viewport MVE head) + `ml/.../training/refine.py` (rewritten) + `ml/.../training/eval_v2.py` (4-family reconstruction) | Reconstruction across **all four** families: Burning Ship in the differentiable renderer (subgradient via `torch.abs`), viewport regression head (always-on, 4 means + 4 log-vars), softplus-parameterised viewport refinement that guarantees `x_max > x_min` and `y_max > y_min` without barriers. Mandelbrot and Burning Ship refine viewport; Julia and Multibrot can optionally co-refine viewport with their family-internal parameter | same |
| **Slice 4c** | `ml/.../models/cnn_v2.py` (embedding head) + `ml/.../training/contrastive.py` + `ml/.../embeddings.py` | Embedding space + retrieval: L2-normalised projection head, supervised contrastive (SupCon) loss with family labels as positives, `embeddings.npz` bank (`build-embeddings`), top-K cosine-similarity retrieval (`retrieve`) with optional family filter and recipe-row lookup | same |

## Stage 8 — the Android client

The starting `app/` was a single Java `MainActivity` with an XML layout. Stage 8 swaps the lot for Kotlin + Jetpack Compose and wires it to both the Java backend and the Python ML service through one HTTP client.

### Stack choices

| Concern | Pick | Why |
|---|---|---|
| UI | **Jetpack Compose + Material 3** | Declarative, idiomatic on Android 2024+, no XML layout layer to maintain |
| Networking | **Ktor 3 (OkHttp engine) + kotlinx.serialization** | Native-Kotlin suspending API, no annotation processing, the `JsonClassDiscriminator` feature lines up cleanly with the backend's external `fractalType` discriminator |
| Image loading | **Coil 3** | Compose-native, accepts `ByteArray` directly so we don't temp-file the base64-decoded PNG |
| Architecture | Repository → ViewModel → Compose, manual DI through `Application` | Two screens + one HTTP service is below the threshold where Hilt/Dagger pays for itself |
| Build | Kotlin 2.0.21, AGP 8.13.1, JDK 17 toolchain, compileSdk 36 | Compose plugin needs Kotlin 2.0 + a 17+ source level |

### Module layout

```
app/src/main/java/com/example/myapplication/
├── FractalovApp.kt              Application subclass = manual DI root (singleton FractalovApi)
├── MainActivity.kt              ComponentActivity + Compose NavHost
├── domain/
│   ├── FractalRecipe.kt         @Serializable mirror of backend's wire shape
│   └── Presets.kt               Default recipes per family
├── network/
│   ├── Dto.kt                   Request/response DTOs
│   └── FractalovApi.kt          Ktor client, render() + mlRenderFromImage()
└── ui/screen/
    ├── HomeScreen.kt            Pick family → render → preview
    ├── HomeViewModel.kt
    ├── MlScreen.kt              Pick photo → /ml/render-from-image → preview
    └── MlViewModel.kt
```

### Cross-language wire compatibility

The most non-trivial bit is the `FractalParams` polymorphism, because the same payload travels:

```
kotlinx.serialization (client)  ──HTTP──►  Jackson @JsonTypeInfo (backend)
                                         ──HTTP──►  pydantic-style FastAPI (ML)
```

All three sides use the recipe-level `fractalType` field as the discriminator:

* **Backend (Jackson)**: `@JsonTypeInfo(include = EXTERNAL_PROPERTY, property = "fractalType")` on `FractalRecipe.params` — the discriminator lives outside the polymorphic object.
* **Client (kotlinx.serialization)**: `@JsonClassDiscriminator("fractalType")` on the sealed `FractalParams`, with `classDiscriminatorMode = POLYMORPHIC`. Together these put the discriminator at the recipe level on serialise and read it from there on deserialise — same wire shape Jackson expects.
* **ML (pydantic)**: doesn't deserialise `FractalParams` at all — it just passes the recipe back to the backend as a JSON dict, keeping the discriminator intact.

Result: the same JSON travels through three runtimes with no per-language schema. The single test that proves this is `recipeJsonRoundTripsThroughJsonbColumn` in `backend/src/test/java/.../persistence/ProjectAndRecipeIT.java` (Stage 3).

### Configuration

Backend URL is a `BuildConfig` field set in `app/build.gradle.kts`:

```kotlin
buildConfigField("String", "BACKEND_BASE_URL", "\"http://10.0.2.2:8080\"")
```

`10.0.2.2` is the AVD alias for the host's `localhost`, so an emulator running on the same machine as the backend works out of the box. For a device on the LAN, change the URL to `http://<host-lan-ip>:8080` or wire it through a debug-only setting.

Cleartext HTTP is allowed via `app/src/main/res/xml/network_security_config.xml`, but only for `10.0.2.2 / localhost / 127.0.0.1` — production builds will need HTTPS.

### Two screens

* **Home** — four `FilterChip`s for the families, one Render button. Tapping fires `FractalovApi.render(preset)` on a coroutine; on success Coil renders the returned PNG bytes directly. This is the smoke test for the whole stack: client serialisation → backend deserialisation → render → base64 → client decode → display.

* **ML from image** — `PickVisualMedia` photo picker → `FractalovApi.mlRenderFromImage(bytes, filename)` → backend `/ml/render-from-image` → Python ML service classifies + predicts c → backend re-renders → client downloads the persisted PNG via the imageUrl from the response. Six hops, one button.

Each screen has a tiny `ViewModel` holding a `MutableStateFlow<UiState>` with idle/loading/done/error sealed states. No DataStore, no Room — Stage 8 is about integration, not local persistence.

### Build it

```bash
# Pin JDK 17 in root gradle.properties (one-time, already done in the repo)
# org.gradle.java.home=/Library/Java/.../ms-17.0.18/Contents/Home

./gradlew :app:assembleDebug          # produces app-debug.apk (~21 MB)
./gradlew :app:installDebug           # installs on a connected device/emulator
```

Make sure the backend is running on `:8080` and (for the ML screen) the Python service is on `:9000` before opening the app — both are documented in `backend/README.md` and `ml/README.md`.

### What the Android client implements **today**

| Screen | What it does | Backend hops |
|---|---|---|
| `HomeScreen` | Pick one of four families via `FilterChip`, tap **Render**, see the preset render | `POST /render` → base64 PNG → Coil |
| `MlScreen` | `PickVisualMedia` photo picker → upload → see the ML-suggested render with family + confidence + (Julia) c | `POST /ml/render-from-image` → follow `imageUrl` |

Two `ViewModel`s, one `FractalovApi` (Ktor + OkHttp + kotlinx.serialization),
manual DI through a singleton `Application` subclass. Renders come back as
`ByteArray` and are fed to Coil 3 directly — no temp files, no
`BitmapFactory`.

### What is **not yet** in the Android client

The Slice 3 / 4b / 4c research surface is still backend-only. The
following five screens will turn the app from a "look at one render"
demo into a research interface — see the planning section below.

- **No predicted-vs-original side-by-side.** Slice 3 can compare
  pure-NN vs hybrid-refined renders, but the client only shows the
  final render; no oracle to compare against on screen.
- **No uncertainty visualisation.** The Slice 3 model returns
  `(μ, σ)` for Julia c and Multibrot exponent; the existing
  `/ml/suggest-from-image` endpoint surfaces only `μ`.
- **No variations grid.** Backend has `/ml/variations`; nothing in
  Compose calls it yet.
- **No retrieval (embedding-similar).** Slice 4c built the embedding
  bank + retrieval CLI; no `/ml/similar` endpoint yet, no UI.
- **No recipe-vs-recipe compare.** No client-side ablation tool; the
  backend's stateless `/render` endpoint is the only thing needed.
- **No history / projects screen.** Stage 3 endpoints are there
  (`GET /projects`, `GET /recipes/{id}/renders`), but the screens
  that browse them belong in a follow-up.
- **No async-job UI.** Stage 4 added `/recipes/{id}/render-jobs` +
  SSE; the client still uses the synchronous `/render` for previews.
- **No HTTPS / production config.** Cleartext to dev hosts only.
- **No instrumentation tests.** The `androidTest/` skeleton is there;
  meaningful Compose UI tests would be a separate slice.

## What's the whole point

* **Stages 1-4** — a credible Java/Spring backend that anyone reviewing the repo can stand up locally. Postgres, async jobs, SSE — proper engineering rather than a notebook with embedded math.
* **Stages 5-7** — a credible ML pipeline that uses the backend itself as a data source, trains a small CNN with proper Optuna tuning, and serves it back as an inference HTTP service the backend consumes.
* **Stage 8** — proves the loop closes: an actual user-facing client speaking the same protocols.
* **Slices 0–4** — turn the system into a research artifact: a generalised
  rendering framework (one engine, four kernels behind one interface), a
  benchmark harness with per-stage timing, a frozen synthetic dataset with a
  manifest and five splits including adversarial ones, multi-task inverse
  inference with calibrated uncertainty + hybrid NN+L-BFGS reconstruction
  for all four families, supervised-contrastive embeddings + retrieval bank.

For the research write-up:

* `backend/README.md` documents per-renderer perf baselines that become the
  **before** numbers in the Slice 1 benchmark matrix.
* `ml/README.md` documents acceptance numbers for the Stage 6 baseline; the
  Slice 3 v2 pipeline reports the **after** numbers per split (test /
  near_ood / hard_overlap), each with calibration σ-coverage and pre→post
  reconstruction MSE for each of the four families.

## What is implemented right now

Three independent runtimes (backend / ml / app) plus four research slices on
top of the original eight stages. Concrete state:

### Backend (Java 21, Spring Boot 3)

- Stateless `/render` (Stage 1–2) and persisted `/recipes/{id}/renders`
  (Stage 3) for all four families.
- Async `/render-jobs` with `FOR UPDATE SKIP LOCKED` + SSE (Stage 4),
  multi-instance-safe via a `claim_token` UUID column (Slice 0).
- ML proxy under `/ml/*` (Stage 7).
- Generalised escape-time engine + four `IterationKernel` implementations
  + cardioid/period-2 short-circuits for Mandelbrot (Slice 0).
- Distance-estimation channel + `DISTANCE_ESTIMATE` colour mode for
  Mandelbrot/Julia/Multibrot (Slice 0).
- `/bench/{compare, scenarios, matrix, matrix-from-presets}` gated by
  `app.bench.enabled` + `backend/scripts/run-matrix.sh` for CSV pivoting
  (Slice 1).
- Test count: **98 passing** across renderer correctness, persistence,
  async jobs, the matrix benchmark controller, and a Vector-API parity
  test against the optimised Mandelbrot kernel.

### ML (Python 3.13, PyTorch 2.x)

- Backend client + recipe sampler + parquet `Dataset` (Stage 5).
- Two-headed v1 FractalCNN + Optuna tuning + reconstruction (Stage 6).
- FastAPI inference server (`fractalov-ml serve`) + Java `MlClient`
  (Stage 7).
- Config-driven dataset (`build-dataset` CLI, `research-v1.yaml`,
  five splits, manifest with SHA-256 over every PNG + the parquet) — Slice 2.
- Multi-task v2 model (`FractalCNNv2`) with MVE Gaussian-NLL heads for
  Julia c, Multibrot exponent, and viewport — Slices 3 + 4b.
- Differentiable PyTorch renderer for all four families (Mandelbrot,
  Julia, Multibrot, Burning Ship) with tensor-viewport gradients — Slice 4b.
- Hybrid NN-warm-start + L-BFGS reconstruction refiner with softplus
  viewport parameterisation — Slice 4b.
- `train-v2` / `eval-v2` CLI: per-epoch metrics, per-split eval
  (test/near_ood/hard_overlap), σ-coverage calibration, pre→post
  reconstruction MSE per family — Slice 3 + 4b.
- `tune-v2` CLI: Optuna search over lr / weight_decay / batch_size /
  c_weight / exp_weight / vp_weight / base_channels / augmentation —
  Slice 4a.
- Embedding head (L2-normalised projection MLP) + supervised
  contrastive loss + `build-embeddings` / `retrieve` CLI — Slice 4c.
- Test count: **96 passing** (offline; no backend needed).

### Android client (Kotlin 2.0, Jetpack Compose, Ktor 3)

- Two screens: family-preset render, ML-from-image (Stage 8).
- One `FractalovApi` (Ktor + OkHttp + kotlinx.serialization), manual
  DI via `Application`, Coil 3 for `ByteArray` PNG rendering.
- The Slice 3 / 4b / 4c research surface is **not yet wired into the
  client** — that is the next slice.

## Frontend plan (next slice)

Five screens turn the app from a "render one preset" demo into a
research interface. Planned in priority order; each is a 1-day slice
(150–300 lines of Compose + ViewModel + Ktor wiring).

| # | Screen | Backend hops needed | What it shows |
|---|---|---|---|
| 1 | **Predicted vs reconstructed side-by-side** | existing `/ml/render-from-image`; show original alongside the reconstructed render | The most direct visualisation of inverse inference quality |
| 2 | **Variations grid** | existing `/ml/variations`; render thumbnails of N perturbed recipes | Lets the user explore the local manifold around an ML-suggested recipe |
| 3 | **Uncertainty visualisation** | new `/ml/suggest-from-image-v2` returning `(μ, σ)` and family distribution; replace existing `/ml/suggest-from-image` or add a flag | Shows σ as error-bars on c / exponent; calibration + family confidence as a distribution bar |
| 4 | **Embedding-similar (retrieval)** | new `/ml/similar?file=...&k=N` endpoint backed by the Slice 4c bank; returns top-K recipes + thumbnails | The most "research-y" sceenario — the embedding space has visual meaning |
| 5 | **Recipe-vs-recipe compare** | uses existing `/render` twice; pure client-side state | Side-by-side ablation tool; useful for benchmarking changes |

The first two need **zero** backend changes; the remaining three need
small additive endpoints under `/ml/*`. None of these affect the
existing render / recipe / job / embedding pipelines.

## Common operations cheat-sheet

```bash
# 1. Local Postgres (one-time)
brew services start postgresql@16
psql -h localhost -d postgres <<'SQL'
CREATE ROLE fractalov WITH LOGIN PASSWORD 'fractalov_dev' CREATEDB;
CREATE DATABASE fractalov_dev OWNER fractalov;
SQL

# 2. Backend (terminal 1)
cd backend && ./gradlew bootRun
# Or, with the benchmark matrix endpoints enabled (Slice 1):
cd backend && FRACTALOV_BENCH_ENABLED=true ./gradlew bootRun

# 3. ML — full research workflow (Slices 2–4c)
cd ml && uv sync --extra dev

# 3a. Build the frozen-artifact dataset (Slice 2)
uv run fractalov-ml build-dataset \
    --config configs/dataset/research-v1.yaml \
    --output-dir data/research-v1

# 3b. Tune v2 hyperparameters (Slice 4a) — produces best_params_v2.json
uv run fractalov-ml tune-v2 \
    --data-root data/research-v1 \
    --out-dir runs/v2-tune \
    --n-trials 15 --epochs-per-trial 6

# 3c. Train v2 with best hyperparameters + supervised-contrastive embedding head
uv run fractalov-ml train-v2 \
    --data-root data/research-v1 \
    --out-dir runs/v2-best \
    --from-best runs/v2-tune \
    --epochs 30 --contrastive-weight 0.5

# 3d. Eval per split + hybrid reconstruction MSE for all four families (Slice 4b)
uv run fractalov-ml eval-v2 \
    --data-root data/research-v1 \
    --checkpoint runs/v2-best/best.pt \
    --out-dir runs/v2-best

# 3e. Build the retrieval embedding bank, then query it (Slice 4c)
uv run fractalov-ml build-embeddings \
    --data-root data/research-v1 \
    --checkpoint runs/v2-best/best.pt
uv run fractalov-ml retrieve \
    --bank data/research-v1/embeddings.npz \
    --checkpoint runs/v2-best/best.pt \
    --image my_query.png \
    --data-root data/research-v1 --k 10

# 3f. Inference HTTP server (only needed for the Android /ml/* path)
uv run fractalov-ml serve --checkpoint runs/v2-best/best.pt --port 9000

# 4. Backend benchmark matrix → CSV (Slice 1)
backend/scripts/run-matrix.sh research 3 10

# 5. Android client (terminal 3 — emulator running with ports forwarded automatically)
./gradlew :app:installDebug
```
