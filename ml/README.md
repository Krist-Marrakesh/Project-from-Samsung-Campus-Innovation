# fractalov-ml

Python ML pipeline for the fractalov project. Three concerns in one package:

* **Stage 5** — dataset generation. Drive the Java backend's `/render` endpoint with randomised recipes, capture the resulting PNGs and ground-truth labels into a parquet, expose the result through a PyTorch `Dataset`.
* **Stage 6** — CNN baseline. Two-headed FractalCNN (family classifier + Julia c regression). Training, Optuna tuning, evaluation, reconstruction.
* **Stage 7** — inference server. FastAPI app loaded with a trained checkpoint, called by the Java backend over HTTP for ML-assisted recipe suggestion.

The Python project is **standalone** — same pattern as `backend/`: own `pyproject.toml`, own venv, never wired into Gradle. Coordination with the rest of the system happens over HTTP only, so backend and ML iterate independently.

## Setup

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/) (one-line install: `pip install uv`). The lockfile pins everything else.

```bash
cd ml
uv sync --extra dev          # creates .venv, installs runtime + test deps
uv run pytest                # 13 offline tests (no backend needed)
uv run fractalov-ml --help   # CLI entry point (typer)
```

`pyproject.toml` lists the runtime stack: `httpx + numpy + pandas + pyarrow + scikit-learn + torch + torchvision + pillow + typer + rich`. PyTorch ships the CPU build by default; switch the index URL in `pyproject.toml` if you want CUDA wheels later.

## Workflow at a glance

```
┌──────────────────┐    POST /render     ┌──────────────────┐
│  recipes.py      │  ─────────────────► │  Java backend    │
│  (random sampler)│  ◄─── PNG + perf ── │  (Stages 1-2)    │
└──────────────────┘                     └──────────────────┘
        │                                          ▲
        │ writes                                   │
        ▼                                          │
   data/<run>/                                     │
   ├── images/<family>/*.png                       │
   ├── labels.parquet                              │
   ├── splits.parquet (train/val/test)             │
   └── generation_meta.json                        │
        │                                          │
        ▼                                          │
   FractalDataset (PyTorch)  ──► Stage 6 training ─┘
```

## Commands

The CLI exposes two parallel pipelines that **share the inference server but
are NOT interchangeable upstream**:

> ⚠️ **v1 (Stage 5/6) and v2 (Slice 2/3/4) datasets have different on-disk
> layouts and different training loops.** Picking the wrong combination
> produces a confusing schema error rather than a silent miss-train.
>
> | Pipeline | Dataset cmd | Layout | Train cmd | Compatible? |
> |---|---|---|---|---|
> | **v1 (legacy)** | `generate` + `split` | `labels.parquet` + separate `splits.parquet` | `train` / `tune` / `eval` / `reconstruct` | ❌ does not work with v2 trainer |
> | **v2 (current research)** | `build-dataset` | single `labels.parquet` with `split` column + `dataset-manifest.json` | `train-v2` / `tune-v2` / `eval-v2` | ❌ does not work with v1 trainer |
>
> The two share `serve` (FastAPI inference, Stage 7) only because both
> emit the same `FractalCNN`-shaped checkpoint format. The Slice 4
> uncertainty heads on `FractalCNNv2` need a v2 checkpoint at the
> serving side; the current `serve` command still loads v1 by default.

### `generate` — call the backend, build a dataset

```bash
# Backend must already be on :8080 (`cd backend && ./gradlew bootRun`)
uv run fractalov-ml generate \
  --output-dir data/v1 \
  --per-family 5000 \
  --width 128 --height 128 \
  --seed 42
```

Knobs:

| Flag             | Default                  | Effect                                                              |
|------------------|--------------------------|---------------------------------------------------------------------|
| `--output-dir`   | required                 | Where the run lands. Subdirs `images/<family>/` are created.        |
| `--per-family`   | `1000`                   | Examples per family. Total = `per_family * len(families)`.          |
| `--seed`         | `42`                     | RNG seed; identical seeds reproduce the recipe stream byte-for-byte.|
| `--backend-url`  | `http://localhost:8080`  | Override if the backend lives elsewhere.                            |
| `--width / --height` | `128`                | Image dimensions in pixels. The backend handles up to 8192².        |
| `--ssaa`         | `1`                      | Backend `samplesPerAxis` (1..3). Higher = smoother but ~N² slower.  |
| `--family`       | all four                 | Repeat to subset, e.g. `--family julia --family mandelbrot`.        |

Sampling strategy is in `recipes.py`. Mandelbrot/Burning-Ship use viewport perturbations around their classic windows; Julia samples `c` uniformly inside the disk `|c| ≤ 1.5`; Multibrot exponent ∈ `{2..6}`. Pure noise over `[-2,2]²` would put the camera in trivial all-escape regions for most samples.

### `split` — stratified train/val/test

```bash
uv run fractalov-ml split \
  --labels data/v1/labels.parquet \
  --out    data/v1/splits.parquet \
  --val-size 0.1 --test-size 0.1
```

Stratified on `family` so every split contains every family in proportion. The output is a separate parquet (`id, split` columns) — `labels.parquet` is never modified, which lets you experiment with multiple split policies on the same labels.

### `inspect` — EDA tables

```bash
uv run fractalov-ml inspect --root data/v1
```

Prints, in order: family distribution, per-family numeric param ranges (median ± IQR), per-stage render times (these will be the **before-numbers** for Stage 10 perf comparisons), and split counts if `splits.parquet` is present.

## Schemas

### `labels.parquet`

One row per successfully rendered example.

| column              | dtype     | notes                                                  |
|---------------------|-----------|--------------------------------------------------------|
| `id`                | str       | UUID hex; matches `image_path` filename                |
| `family`            | str       | one of `mandelbrot/julia/burning_ship/multibrot`       |
| `fractal_type`      | str       | mirrors `family` (kept for forward compat with backend)|
| `c_re`, `c_im`      | float64   | populated for **julia** only; `NaN` elsewhere          |
| `exponent`          | float64   | populated for **multibrot** only; `NaN` elsewhere      |
| `max_iter`          | int64     | iteration budget                                       |
| `escape_radius`     | float64   | always `2.0` in the default sampler                    |
| `smoothing`         | bool      | always `True` in the default sampler (richer signal)   |
| `vp_x_min/x_max/y_min/y_max` | float64 | viewport bounds                              |
| `palette`           | str       | one of `grayscale/fire/ocean/rainbow_cyclic`           |
| `color_mode`        | str       | `linear` or `histogram`                                |
| `samples_per_axis`  | int64     | backend SSAA factor                                    |
| `image_path`        | str       | relative to dataset root, e.g. `images/julia/abc.png`  |
| `perf_render_ms`/`colorize_ms`/`encode_ms`/`total_ms` | int64 | per-stage timing reported by the backend |
| `file_size_bytes`   | int64     | PNG byte length                                        |
| `request_id`        | str       | backend-side correlation id                            |

### `splits.parquet`

Two columns only — `id, split` where `split ∈ {"train","val","test"}`. Joined to `labels.parquet` on `id` by the dataset class.

### `generation_meta.json`

Run-level metadata: `seed`, `backend_url`, `per_family`, `families`, `sample_config`, `successful`, `failed`. Anything we'd need to repro the run later.

## PyTorch Dataset

```python
from pathlib import Path
from torch.utils.data import DataLoader
from fractalov_ml.dataset import FractalDataset, FamilyEncoder

train = FractalDataset(Path("data/v1"), split="train")
loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=4)

for images, family_label, params in loader:
    # images: (B, 3, H, W) float32 in [0, 1]
    # family_label: (B,) int64 in [0..3] — alphabetic by family name
    # params: dict of batched tensors:
    #   c_re, c_im       — float, NaN where the family doesn't have one
    #   exponent         — long, -1 where the family doesn't have one
    #   has_c, has_exp   — bool masks for the two above (use them as loss mask)
    #   max_iter, vp_x_min, ...
    ...

# Decode predictions back to the family name via the stable encoder.
enc = FamilyEncoder.default()
print(enc.decode(int(family_label[0])))  # e.g. "julia"
```

The encoder's class ordering is **alphabetic** (`burning_ship=0, julia=1, mandelbrot=2, multibrot=3`), not parquet-order. That keeps a model trained on dataset A directly usable on dataset B even if the family ratios differ.

Family-specific params (`c_re`, `c_im`, `exponent`) use sentinel values (`NaN` / `-1`) instead of `None` so PyTorch's default collate can batch them. The boolean `has_c` / `has_exponent` flags double as loss masks: `loss = ((pred - target) ** 2 * has_c).mean()`.

## Smoke test results (Stage 5 acceptance)

100 examples (25 × 4 families), 96×96, smoothing on, default palettes:

```
backend wall-clock for the whole run         < 1 second
labels.parquet                               ~ 24 KB
images/                                      4 × 25 PNGs (~10–14 KB each)
EDA: balanced 25/25/25/25, splits 80/10/10   stratified per-family
DataLoader batch shape                       (16, 3, 96, 96) float32 in [0, 1]
```

Multibrot was the slowest family — atan2/pow per iteration — exactly matching the Stage 2 baseline. That signal will be tracked across Stage 10 backend optimisations.

## Stage 6: CNN baseline + Optuna tuning

A two-headed ResNet-style CNN that, from a 3-channel fractal image, predicts:

1. **Which family produced it** (4-way softmax classifier)
2. **For Julia, the value of `c = (c_re, c_im)`** (2-D regression with masked loss)

The Julia head is masked: only `julia` rows contribute to its gradient (via the `has_c` flag from the dataset). Mandelbrot/Burning-Ship/Multibrot rows are filtered out by index-select **before** the loss is computed — multiplying by a 0/1 mask after the fact propagates NaN if the regression head ever produces non-finite activations.

### Architecture

`FractalCNN` lives in `src/fractalov_ml/models/cnn.py`:

```
input (B, 3, H, W)
  ↓ Conv 3x3 (3→base) + GroupNorm + ReLU + MaxPool 2x2
  ↓ ResBlock (base→2*base, stride 2)
  ↓ ResBlock (2*base→4*base, stride 2)
  ↓ ResBlock (4*base→8*base, stride 2)
  ↓ AdaptiveAvgPool(1) + Flatten
  ├─→ Linear (8*base → 4)                          family_logits
  └─→ Linear (8*base → 64) + ReLU + Linear (64→2)
      + Tanh × 1.5                                 c_pred ∈ [-1.5, 1.5]²
```

`base_channels` is tunable (`{16, 32, 48}` in the search space). Pixel input is resolution-agnostic thanks to `AdaptiveAvgPool` — same checkpoint runs on 96×96 or 256×256 without surgery.

**Why GroupNorm, not BatchNorm.** BatchNorm tracks running mean/var statistics that, on PyTorch 2.x MPS, can desync between train and eval modes — a checkpoint that hits 0.93 val accuracy in `train.py` reloads as a 0.25 trivial classifier in `eval.py`. GroupNorm has no running buffers and produces identical numerics on every device. The cost is a small constant factor in throughput; the win is checkpoints that actually work across devices and runs.

### Training

```bash
# 1. Generate a dataset (once; reused by all training runs).
uv run fractalov-ml generate --output-dir data/v1 --per-family 500 \
  --width 128 --height 128 --seed 42
uv run fractalov-ml split --labels data/v1/labels.parquet \
  --out data/v1/splits.parquet

# 2. Optional: tune hyperparameters with Optuna.
uv run fractalov-ml tune --data-root data/v1 --out-dir runs/tune \
  --n-trials 10 --epochs-per-trial 6 --device cpu

# 3. Train. --from-best reads runs/tune/best_params.json automatically.
uv run fractalov-ml train --data-root data/v1 --out-dir runs/best \
  --epochs 25 --from-best runs/tune --device cpu

# 4. Evaluate on the held-out test split.
uv run fractalov-ml eval --data-root data/v1 \
  --checkpoint runs/best/best.pt --out-dir runs/best --device cpu

# 5. Reconstruction quality: predict c, re-render, compare to original.
#    Requires the backend on :8080 (Stage 1 endpoint /render).
uv run fractalov-ml reconstruct --data-root data/v1 \
  --checkpoint runs/best/best.pt --out-dir runs/best --device cpu \
  --max-samples 100
```

Outputs in `runs/best/`:

| file | content |
|---|---|
| `best.pt` | state dict at the lowest val *cls_loss* — not joint loss; see below |
| `last.pt` | full checkpoint dict at the last completed epoch (resume-friendly) |
| `metrics.json` | per-epoch history (lr, time, train + val loss/acc/c_mse) |
| `config.json` | the exact `TrainingConfig` that produced this run |
| `test_metrics.json` | family acc + confusion + Julia c MAE/MSE |
| `reconstruction_metrics.json` | mean/median/p90 pixel MSE for re-renders |

**Best-checkpoint criterion is val cls_loss, not joint loss.** The joint loss is dominated by the regression head, which collapses to ~0 quickly because non-julia rows contribute zero. A model that scored low joint loss by getting reg=0 while predicting nothing useful for classification would beat a genuinely good model on joint loss. Tracking val_cls_loss aligns the saved checkpoint with the head we actually care about for the downstream classifier.

### Hyperparameter tuning (Optuna)

`fractalov-ml tune` runs a TPESampler search with MedianPruner over:

| param | range | scale |
|---|---|---|
| `learning_rate` | `1e-5 .. 5e-3` | log |
| `weight_decay` | `1e-6 .. 1e-2` | log |
| `batch_size` | `{32, 64, 128}` | categorical |
| `reg_weight` (β in `α·cls + β·reg`) | `0.1 .. 20.0` | log |
| `base_channels` | `{16, 32, 48}` | categorical |
| `enable_augmentation` | `{False, True}` | categorical |

Each trial is a short training run (default 6–10 epochs) that reports val loss back as an intermediate value. MedianPruner kills trials whose val loss is below median after a 2-epoch warmup. Trials that hit a non-finite loss (numerical instability with bad lr/β picks) are caught and re-classified as pruned — the parameter combo is unviable but doesn't fail the whole study. Outputs:

- `runs/tune/best_params.json` — pass to `train --from-best`
- `runs/tune/study_summary.json` — ranked trial summary
- `runs/tune/trial_NNN/` — per-trial checkpoint + metrics, so a successful trial doubles as a usable model

### Acceptance numbers (Stage 6, 2 000 images, 128×128)

Hyperparameters chosen by Optuna (10 trials × 6 epochs):

```
learning_rate = 4.15e-3   weight_decay = 2.14e-3   batch_size = 32
reg_weight = 0.50         base_channels = 16        augmentation = off
```

Full run (25 epochs, ≈3 minutes on CPU on M4 Max):

| metric | value |
|---|---|
| **train family accuracy** | 0.999 |
| **val family accuracy** | 0.93 |
| **test family accuracy** | **0.93** (50/50 burning_ship · 43/50 julia · 49/50 mandelbrot · 44/50 multibrot) |
| **test Julia c MSE** | **0.044** (MAE_re=0.17, MAE_im=0.15) |
| **reconstruction pixel MSE — median** | **0.0027** |
| **reconstruction pixel MSE — mean** | 0.026 |
| **reconstruction pixel MSE — p90** | 0.057 |
| **all 50 reconstructions succeeded** | yes |

Reconstruction works as the headline result: a model that gets `c_re` to two decimals can still produce a visually-different fractal because the Julia set is non-linear in `c`. Median pixel MSE in `[0, 1]` of `0.0027` means the predicted-`c` re-render is **visually indistinguishable** from the original for half the test set, and the p90 (`0.057`) covers harder cases where the predicted Julia structure is similar but not pixel-perfect.

### Known issue: MPS instability

On Apple Silicon (PyTorch 2.11 + MPS), training this architecture with BatchNorm produced checkpoints whose loaded state didn't match the training-time validation accuracy — a checkpoint that hit 0.93 val accuracy reloaded as a 0.25 trivial classifier (one class for everything). The same code on CPU produced consistent, well-behaved checkpoints.

Switching all normalisation layers to GroupNorm fixed the train↔eval mismatch but didn't fix late-epoch loss explosions on MPS specifically (the c-head's `tanh × 1.5` ouputs occasionally blow up to non-finite values during training, even with gradient clipping). The combined fix:

- **Architecture**: GroupNorm everywhere, no running buffers
- **Training**: `--device cpu` for now (3 minutes vs ~50 seconds on MPS, but correct)
- **Robustness**: a per-batch NaN guard in the train loop drops up to 50 bad batches per epoch silently, then aborts the run

For Stage 10's research-perf comparison, this is itself a useful data point: production training on MPS for `tanh`-bounded heads with mixed-task losses isn't reliable in PyTorch 2.11. Stage 10 should re-test on PyTorch ≥ 2.13 (improved MPS BN/Adam fixes have landed in the nightlies) and on CUDA via a Linux box.

## Stage 7: Inference server

The trained checkpoint is exposed over HTTP through a small FastAPI application. The Java backend is the only consumer; the mobile / web client never talks to the inference server directly.

### Why a separate Python service rather than ONNX-in-Java

For the research phase the ML side will keep changing — new heads, new dataset, new architecture. Re-exporting to ONNX after each iteration adds friction, and PyTorch's MPS-trained `GroupNorm` / custom `Tanh*1.5` heads don't always round-trip cleanly through ONNX yet. A Python inference process keeps the model running as-is. The ~20-50 ms HTTP hop on localhost is invisible next to the actual inference + render time, and the swap to in-process ONNX (Stage 10 candidate, once architecture stabilises) is a drop-in change behind the same Java `MlClient` surface.

### Run it

```bash
uv run fractalov-ml serve \
  --checkpoint runs/best/best.pt \
  --port 9000 \
  --device cpu
```

Endpoints (FastAPI, OpenAPI docs at `/docs`):

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/healthz` | liveness probe — also reports the loaded checkpoint path, device, and `base_channels` so the operator knows what's actually serving |
| `POST` | `/infer/suggest-from-image` | multipart PNG upload → `Suggestion {family, family_confidence, family_distribution, c_re?, c_im?, recipe}` |
| `POST` | `/infer/variations` | JSON `{recipe, count, seed}` → list of perturbed recipes (rule-based, no model) |

The recipe coming back from `/infer/suggest-from-image` is **wire-compatible** with the Java backend's `FractalRecipe` — pass it straight to `/render` to materialise the picture.

### Java side (Stage 7 backend)

The Java backend has three new endpoints under `/ml/`. They proxy to the Python service and (for the render flow) tie the result into the existing project / recipe / render persistence stack:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ml/suggest-from-image` | multipart upload → returns the suggestion verbatim. No DB writes. |
| `POST` | `/ml/render-from-image` | upload → suggestion → save the predicted recipe under an auto-created bucket project (`"ML suggestions YYYY-MM-DD"`, owner `ml`) → run the existing `RenderHistoryService` to render+persist → return the render record with `imageUrl`. One HTTP call replaces ~5 manual ones. |
| `POST` | `/ml/variations` | JSON body, proxied to the Python `/infer/variations` |

If `app.ml.enabled=false` (env: `FRACTALOV_ML_ENABLED=false`) or the Python server is unreachable, every `/ml/*` endpoint returns **HTTP 503 `ml_unavailable`** with a descriptive message — nothing in the rest of the backend depends on ML being up, so a model outage degrades cleanly.

### Tunables (env vars)

| Variable | Default | Effect |
|---|---|---|
| `FRACTALOV_ML_ENABLED` | `true` | Master toggle. `false` makes `/ml/*` return 503 without trying. |
| `FRACTALOV_ML_URL` | `http://localhost:9000` | Where the Python server lives. |
| `FRACTALOV_ML_TIMEOUT_MS` | `10000` | Per-request timeout. |

### End-to-end walkthrough

```bash
# Terminal 1 — Java backend (Stages 1-4 + 7)
cd backend && ./gradlew bootRun

# Terminal 2 — Python ML server
cd ml && uv run fractalov-ml serve --checkpoint runs/best/best.pt --port 9000

# Terminal 3 — exercise the integrated /ml/* surface
SAMPLE=ml/data/v1/images/julia/$(ls ml/data/v1/images/julia | head -1)

# Pure suggestion (no DB writes, no rendering)
curl -F "file=@$SAMPLE" localhost:8080/ml/suggest-from-image | jq

# Suggest + render + persist (creates project / recipe / render rows)
RES=$(curl -s -F "file=@$SAMPLE" localhost:8080/ml/render-from-image)
echo "$RES" | jq '{family: .suggestion.family, projectId, recipeId, imageUrl: .render.imageUrl}'

# Download the persisted PNG via the existing Stage 3 endpoint
RID=$(echo "$RES" | jq -r .render.id)
curl -s -o /tmp/ml.png "localhost:8080/renders/$RID/image" && open /tmp/ml.png

# Generate a few visual variations of an existing recipe
curl -s -X POST localhost:8080/ml/variations \
  -H 'Content-Type: application/json' \
  -d '{"recipe":{"viewport":{"xMin":-1.5,"xMax":1.5,"yMin":-1.5,"yMax":1.5},
       "renderSettings":{"widthPx":256,"heightPx":256,"samplesPerAxis":1},
       "colorSettings":{"paletteName":"fire","mode":"linear"},
       "fractalType":"julia",
       "params":{"cRe":-0.7,"cIm":0.27,"maxIter":200,"escapeRadius":2.0,"smoothing":true}},
       "count":3,"seed":42}' | jq
```

### One non-obvious wire detail

The Java backend uses the JDK `HttpClient` for the multipart upload to the Python server. We pin `HttpClient.Version.HTTP_1_1` explicitly: by default the JDK adds an `Upgrade: h2c` header trying to negotiate HTTP/2 over cleartext, which uvicorn doesn't speak — FastAPI then sees the body as part of the post-upgrade switch and discards it, surfacing as a confusing 422 "Field required: file". Pinning HTTP/1.1 makes the request shape diff-able against `curl` and removes the surprise. Documented in `MlClient`'s class comment for whoever hits this next.

## What's not in scope yet

- **No raw escape-time output** — the backend returns coloured PNG. Stage 8+ may add a `/render-raw` endpoint if the regression head benefits from the unprocessed `double[][]` signal.
- **No multi-task ablations beyond family + Julia c** — Stage 8 picks up Multibrot exponent regression and Burning-Ship subspace probes.
- **No CUDA / Linux benchmarking** — comparison numbers across devices is a Stage 10 task once we know the architecture is final.
- **No ONNX export / in-process inference** — Stage 10 candidate once the architecture stabilises and the inference HTTP hop becomes worth removing.
- **No streaming / batching of inference requests** — the Python server processes one image per request. Adequate for interactive ML-assisted suggestion; would need rework for high-QPS workloads (none exist in this project yet).
