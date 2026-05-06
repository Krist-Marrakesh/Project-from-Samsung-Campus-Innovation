# fractalov — status & roadmap

Snapshot of what's built, what works, and what's intentionally
deferred. Lives next to `docs/research-report.md` so a reviewer can
read both in one sitting.

## At a glance

```
backend/   Java 21, Spring Boot 3        99 tests passing
ml/        Python 3.13, PyTorch          96 fast + 2 slow integration tests passing
app/       Kotlin 2.0, Jetpack Compose   compiles cleanly; on-device fractal renderer
```

Three independent runtimes communicating only over HTTP — never any
direct linking between them. Each has its own toolchain, its own
README, and its own test suite.

## What we have

### Stages 1–8 (the original three-tier system)

| Stage | What's there |
|---|---|
| **1** | Stateless `/render` endpoint, Mandelbrot + Julia, base64 PNG over JSON |
| **2** | BurningShip + Multibrot kernels, supersampling, histogram coloring, per-stage perf breakdown |
| **3** | Postgres + Flyway, projects → recipes → renders, PNGs on the filesystem |
| **4** | Async render-jobs with `FOR UPDATE SKIP LOCKED`, SSE progress streams |
| **5** | Dataset generation against the live backend, parquet labels, PyTorch `Dataset` |
| **6** | Two-headed FractalCNN (family + Julia c), Optuna tuning, eval, reconstruction |
| **7** | FastAPI inference server, Java `MlClient`, `/ml/*` endpoints |
| **8** | Compose UI client (preset render + ML-from-image) |

### Slice 0 — generalised rendering framework

* `EscapeTimeEngine` + `IterationKernel` interface — one parallel walker, four kernel implementations behind one type.
* `FieldsOut`/`FieldStack` — multi-channel output (escape-time + distance-estimate).
* Cardioid + period-2 short-circuits in the production Mandelbrot kernel.
* `DISTANCE_ESTIMATE` colour mode (Mandelbrot/Julia/Multibrot; Burning Ship gracefully falls back to LINEAR).
* `Colorizer` rewritten to use `DataBufferInt` direct writes + quantile LUT, **plus an LRU LUT cache** keyed on a cheap escape-map fingerprint so repeated histogram renders skip the O(N log N) sort.
* `claim_token` UUID column on `render_jobs` (Slice 0's V3 migration) — multi-instance-safe job claiming.
* `/bench/*` endpoints gated by `app.bench.enabled`.

### Slice 1 — benchmark harness as a research artifact

* `StageStats` (per-stage min/max/mean/p50/p90/p99 + raw samples) and `StageBreakdown` (render / colorize / encode / total).
* `BenchmarkPipeline` runs the full pipeline through `warmup + runs`, `BenchmarkScenario` + `BenchmarkPresets` define the matrix axes, `MatrixBenchmarkRunner` walks scenarios sequentially.
* `/bench/{compare, scenarios, matrix, matrix-from-presets}` plus `backend/scripts/run-matrix.sh` to drive it and pivot the JSON into a flat CSV.

### Slice 2 — synthetic dataset as a frozen artifact

* `DatasetConfig` (YAML schema, per-split parameter range overrides, `restrict_to_overrides` for adversarial splits).
* Per-split RNG seed via SHA-256 of the split name — orthogonal streams; reordering splits in YAML doesn't perturb other splits.
* `manifest.write_manifest()` produces `dataset-manifest.json` with `config_sha256`, `parquet_sha256`, `images_root_sha256`.
* `configs/dataset/research-v1.yaml`: five splits — `train`, `val`, `test`, `near_ood`, `hard_overlap`. Burning Ship + DE colour-mode fallback explicitly documented in the YAML.
* CLI: `build-dataset`.

### Slice 3 — inverse inference v2

* `FractalDatasetV2` on the new parquet schema (single `labels.parquet` with embedded `split` column).
* `FractalCNNv2` with multi-task heads: family classifier + Julia c MVE + Multibrot exponent MVE + (Slice 4b) viewport MVE + (Slice 4c) L2-normalised embedding.
* `loss_v2.compute_loss` — masked Gaussian NLL via index-select (NaN-safe by construction).
* `torch_renderer` — differentiable Mandelbrot/Julia/Multibrot/BurningShip in pure PyTorch.
* `training/refine.py` — NN-warm-start + L-BFGS reconstruction refiner.
* `train-v2` / `eval-v2` CLI — per-epoch + per-split (`test` / `near_ood` / `hard_overlap`) eval, σ-coverage calibration, pre→post reconstruction MSE.

### Slice 4a — Optuna for the v2 search space

* `tune_v2.run_study_v2` — Optuna with `TPESampler` + `MedianPruner`. Search space: `lr`, `weight_decay`, `batch_size`, `c_weight`, `exp_weight`, `vp_weight`, `contrastive_weight` (added today), `base_channels`, `enable_augmentation`. Objective is val cls_loss — the same metric `train-v2` uses for `best.pt` selection.

### Slice 4b — reconstruction across all four families

* Burning Ship in `torch_renderer` — non-holomorphic but `torch.abs` provides a subgradient sufficient for L-BFGS.
* `make_grid_from_tensors` — viewport bounds as autograd tensors; the differentiable renderer takes either `ViewportT` (static) or a 4-tuple of tensors.
* Viewport regression head on `FractalCNNv2` (always-on; no mask).
* `_ViewportParams` softplus parameterisation in the refiner — `x_max > x_min` and `y_max > y_min` guaranteed without barrier constraints.
* `eval_v2._reconstruction` runs hybrid refinement on all four families (Mandelbrot/BurningShip use viewport refinement; Julia/Multibrot refine the family-internal parameter, with optional viewport co-refinement).

### Slice 4c — embedding space + retrieval

* `FractalCNNv2.embedding_head` — Linear → ReLU → Linear → L2 normalise.
* `training/contrastive.py` — supervised contrastive loss with NaN-safe `torch.where` gating (the `0 * -inf` bug from the first implementation is regression-tested).
* `embeddings.EmbeddingBank` — `.npz` round-trip, top-K cosine retrieval with optional family filter, `lookup_recipe_rows()` to join hits back to the labels parquet.
* CLI: `build-embeddings`, `retrieve`.

### Android client (Slices 70–84)

* `FractalovTheme` (dark midnight palette + monospace numeric typography), reusable components (`FractalImage`, `ZoomableFractalImage`, `ShimmerBox`, `RecipeBadge`, `FamilySelector`, `StatChip`, `PrimarySurface`, `ErrorState`).
* Animated `AppShell` with single Scaffold + slide+fade NavHost transitions.
* Four screens: Home (preset render + pinch/zoom), Compare (predicted vs reconstructed side-by-side), Variations (parallel-rendered grid + recipe inspector), ML (PickVisualMedia → backend ML pipeline).
* On-device fractal renderer (`render/LocalFractalRenderer.kt`) with all four kernels — eliminates network round-trip during pinch-and-pan, gives sharp pixels at every zoom level. Fast Multibrot via repeated complex multiplication (no `atan2`/`pow`/`cos`/`sin` per iteration). Burning Ship periodicity check for early-exit on in-set points.
* Per-family render resolution policy (Mandelbrot/Julia 1536², Burning Ship 1152², Multibrot 1024²) — keeps each gesture-end render under ~250 ms even on mid-tier devices.
* `CancellationException` is correctly propagated rather than displayed as a "Backend rejected the render" error.

### Cross-cutting

* **Documentation**: `backend/README.md` carries an explicit warning about `/bench/*`. `ml/README.md` carries an explicit warning about v1 vs v2 dataset/training compatibility.
* **Integration test**: `tests/test_train_v2_integration.py` builds a tiny synthetic dataset on the fly, runs `train_v2` for 2 epochs, asserts validation classification loss decreases. Marked `slow` so default `pytest` runs skip it; run with `pytest -m slow`. Catches the entire dataset → model → loss → training-loop chain end-to-end.

## Test coverage

| Suite | Count | Notes |
|---|---|---|
| `backend/` | **99** | Renderer correctness, persistence, async-job integration, benchmark matrix controller, vector-API parity, colorizer + cache contract |
| `ml/` (fast) | **96** | Dataset config, manifest, recipes, model output shapes, MVE math, contrastive math, embeddings, Slice 4b kernels, Optuna search-space coverage |
| `ml/` (slow) | **2** | End-to-end training integration test |

All green as of the latest commit.

## What's left

### Should do soon

| ID | Task | Effort | Why it matters |
|---|---|---|---|
| **#68** | Re-measure backend perf and update `backend/README.md` | medium | Numbers in README predate Slice 0 (cardioid + DataBufferInt). Real-machine `run-matrix.sh` output would replace estimates with live data. Not doable without a running backend instance. |
| **#63** | `build_dataset` retry budget — distinguish 5xx (retry) from 4xx (sampler bug, skip) | small | Currently a transient backend hiccup costs us examples in a long generation run; manifest records the loss but a reviewer wonders why. |
| **#57** | Centralised `FractalFamily` enum across all Python modules | small | `families.py` exists; the migration of `recipes.py` / `recipes_v2.py` / `dataset.py` / `dataset_v2.py` / `build_dataset.py` is half-done. Refactor for cohesion, no functional change. |
| **#60** | Rename `MandelbrotVectorRenderer` → `MandelbrotOptimizedRenderer` | small | Class name is misleading (no SIMD, just cardioid + periodicity). Cosmetic. |

### Bigger pieces, intentionally deferred

| ID | Task | Why deferred |
|---|---|---|
| **#64** | `/infer/serve` v2 — FastAPI inference server loading `FractalCNNv2` with uncertainty heads | Needs a trained v2 checkpoint (none of us have one yet). The CLI exists; `train-v2 → eval-v2 → serve --v2` needs to be run end-to-end on real data first. |
| **#65** | Java backend forwards σ from `/ml/*` through to clients | Depends on #64 (the upstream Python service has to surface σ first) and on the Android uncertainty-visualisation screen we chose not to build. |
| **#66** | `JobEventBus` → Postgres LISTEN/NOTIFY | Single-instance backend works fine with the current in-process bus. Multi-instance is a production concern, not an admissions one. |

### Nice-to-haves we explicitly chose not to do

* Predicted-vs-reconstructed UI on Android (Slice 3 has the data, no UI yet). Reasoning: `Compare` screen does the UX-level version; the research-grade comparison belongs in `docs/research-report.md` with concrete numbers, not in the app.
* Embedding-similar UI on Android (Slice 4c has the bank). Same reason.
* Recipe-vs-recipe compare screen. Useful for ablations, but not the headline.

## Workflow cheat-sheet

```bash
# 1. Backend (Postgres + Spring Boot)
brew services start postgresql@16
psql -h localhost -d postgres <<'SQL'
CREATE ROLE fractalov WITH LOGIN PASSWORD 'fractalov_dev' CREATEDB;
CREATE DATABASE fractalov_dev OWNER fractalov;
SQL
cd backend && ./gradlew bootRun
# (or with the benchmark matrix endpoints enabled)
cd backend && FRACTALOV_BENCH_ENABLED=true ./gradlew bootRun

# 2. ML pipeline (frozen-artifact dataset → tune → train → eval → embeddings)
cd ml && uv sync --extra dev

uv run fractalov-ml build-dataset \
    --config configs/dataset/research-v1.yaml \
    --output-dir data/research-v1

uv run fractalov-ml tune-v2 \
    --data-root data/research-v1 \
    --out-dir runs/v2-tune \
    --n-trials 15 --epochs-per-trial 6

uv run fractalov-ml train-v2 \
    --data-root data/research-v1 \
    --out-dir runs/v2-best \
    --from-best runs/v2-tune \
    --epochs 30 --contrastive-weight 0.5

uv run fractalov-ml eval-v2 \
    --data-root data/research-v1 \
    --checkpoint runs/v2-best/best.pt \
    --out-dir runs/v2-best

uv run fractalov-ml build-embeddings \
    --data-root data/research-v1 \
    --checkpoint runs/v2-best/best.pt
uv run fractalov-ml retrieve \
    --bank data/research-v1/embeddings.npz \
    --checkpoint runs/v2-best/best.pt \
    --image my_query.png \
    --data-root data/research-v1 --k 10

uv run fractalov-ml serve --checkpoint runs/v2-best/best.pt --port 9000

# 3. Backend benchmark matrix → CSV (research artefact)
backend/scripts/run-matrix.sh research 3 10

# 4. Android client (USB-connected device or emulator)
./gradlew :app:installDebug
```

## Tests

```bash
# Backend
cd backend && ./gradlew test

# ML — fast tier (default)
cd ml && uv run pytest

# ML — including the end-to-end integration test
cd ml && uv run pytest -m slow
# or run everything
cd ml && uv run pytest -m 'not slow or slow'
```

---

Last updated alongside the integration-test + Optuna `contrastive_weight`
+ histogram-LUT-cache landings.
