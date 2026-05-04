"""Single typer CLI exposing all fractalov-ml commands.

Stage 5: generate / split / inspect — dataset collection.
Stage 6: train / eval / reconstruct — CNN baseline.
Stage 7: serve — FastAPI inference server consumed by the Java backend.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from . import build_dataset as build_dataset_module
from . import embeddings as embeddings_module
from . import generate as gen_module
from . import inspect as inspect_module
from . import serve as serve_module
from . import splits as splits_module
from .recipes import FAMILIES
from .training.config import TrainingConfig
from .training.config_v2 import TrainingConfigV2
from .training.eval import evaluate as evaluate_fn
from .training.eval import reconstruct as reconstruct_fn
from .training.eval_v2 import evaluate_v2 as evaluate_v2_fn
from .training.train import train as run_training
from .training.train_v2 import train_v2 as run_training_v2
from .training.tune import TuneConfig, load_best_params, run_study
from .training.tune_v2 import TuneConfigV2, load_best_params_v2, run_study_v2

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def generate(
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Where the dataset lives."),
    per_family: int = typer.Option(1000, "--per-family", "-n", help="Examples per family."),
    seed: int = typer.Option(42, "--seed", help="Recipe RNG seed."),
    backend_url: str = typer.Option("http://localhost:8080", "--backend-url"),
    width_px: int = typer.Option(128, "--width"),
    height_px: int = typer.Option(128, "--height"),
    samples_per_axis: int = typer.Option(1, "--ssaa", help="Backend SSAA factor (1..3)."),
    family: list[str] = typer.Option(
        None,
        "--family",
        help=(
            "Restrict generation to the given families. Repeat the flag for several. "
            f"Defaults to all of: {', '.join(FAMILIES)}."
        ),
    ),
) -> None:
    """Generate a labelled dataset against the running backend."""
    console = Console()
    families = list(family) if family else None
    gen_module.generate(
        output_dir=output_dir,
        per_family=per_family,
        seed=seed,
        backend_url=backend_url,
        width_px=width_px,
        height_px=height_px,
        samples_per_axis=samples_per_axis,
        families=families,
        console=console,
    )


@app.command("build-dataset")
def build_dataset(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a YAML dataset config (see configs/dataset/)."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Where the dataset lives."),
    backend_url: str = typer.Option("http://localhost:8080", "--backend-url"),
) -> None:
    """Build a frozen-artifact dataset from a YAML config.

    This is the recommended command for new datasets — it covers the work
    that was previously done by `generate` + `split`, adds the manifest +
    checksum chain, and supports per-split parameter overrides
    (near_ood / hard_overlap). The legacy `generate` and `split` commands
    are still available for back-compat with existing checkpoints.
    """
    console = Console()
    build_dataset_module.build_dataset(
        config_path=config,
        output_dir=output_dir,
        backend_url=backend_url,
        console=console,
    )


@app.command()
def split(
    labels: Path = typer.Option(..., "--labels", help="Path to labels.parquet."),
    out: Path = typer.Option(..., "--out", help="Output splits.parquet path."),
    val_size: float = typer.Option(0.1, "--val-size"),
    test_size: float = typer.Option(0.1, "--test-size"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Create stratified train/val/test split parquet."""
    out_path = splits_module.split(
        labels_path=labels,
        out_path=out,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    Console().print(f"[green]wrote {out_path}")


@app.command()
def inspect(
    root: Path = typer.Option(..., "--root", help="Directory with labels.parquet (and optionally splits.parquet)."),
) -> None:
    """Pretty-print EDA tables for a dataset."""
    inspect_module.inspect_dataset(root)


@app.command()
def train(
    data_root: Path = typer.Option(..., "--data-root", help="Dataset directory."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Where to write checkpoints + metrics."),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(64, "--batch-size"),
    lr: float = typer.Option(1e-3, "--lr"),
    weight_decay: float = typer.Option(1e-4, "--weight-decay"),
    cls_weight: float = typer.Option(1.0, "--cls-weight"),
    reg_weight: float = typer.Option(5.0, "--reg-weight"),
    base_channels: int = typer.Option(32, "--base-channels"),
    patience: int = typer.Option(8, "--patience"),
    device: str = typer.Option(None, "--device", help="mps/cuda/cpu; default = auto-pick."),
    no_aug: bool = typer.Option(False, "--no-aug", help="Disable train-time augmentation."),
    seed: int = typer.Option(42, "--seed"),
    from_best: Path = typer.Option(
        None,
        "--from-best",
        help="Load hyperparameters from a previous tuning run's best_params.json.",
    ),
) -> None:
    """Train the FractalCNN on a generated dataset."""
    if from_best is not None:
        params = load_best_params(from_best)
        # Apply only the tunable knobs; epochs/patience/seed stay CLI-controlled.
        lr = float(params.get("learning_rate", lr))
        weight_decay = float(params.get("weight_decay", weight_decay))
        batch_size = int(params.get("batch_size", batch_size))
        reg_weight = float(params.get("reg_weight", reg_weight))
        base_channels = int(params.get("base_channels", base_channels))
        if "enable_augmentation" in params:
            no_aug = not bool(params["enable_augmentation"])

    cfg = TrainingConfig(
        data_root=data_root,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        cls_weight=cls_weight,
        reg_weight=reg_weight,
        base_channels=base_channels,
        patience=patience,
        device_override=device,
        enable_augmentation=not no_aug,
        seed=seed,
    )
    run_training(cfg)


@app.command()
def tune(
    data_root: Path = typer.Option(..., "--data-root"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    n_trials: int = typer.Option(20, "--n-trials"),
    epochs_per_trial: int = typer.Option(10, "--epochs-per-trial"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option(None, "--device"),
) -> None:
    """Optuna hyperparameter search over the FractalCNN training space."""
    tcfg = TuneConfig(
        data_root=data_root,
        out_dir=out_dir,
        n_trials=n_trials,
        epochs_per_trial=epochs_per_trial,
        seed=seed,
        device_override=device,
    )
    run_study(tcfg)


@app.command()
def eval(
    data_root: Path = typer.Option(..., "--data-root"),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to best.pt or last.pt."),
    out_dir: Path = typer.Option(None, "--out-dir", help="Optional dir to write metrics JSON."),
    device: str = typer.Option(None, "--device"),
) -> None:
    """Evaluate a trained model on the test split."""
    evaluate_fn(data_root=data_root, checkpoint=checkpoint, out_dir=out_dir, device_override=device)


@app.command()
def reconstruct(
    data_root: Path = typer.Option(..., "--data-root"),
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    out_dir: Path = typer.Option(None, "--out-dir"),
    backend_url: str = typer.Option("http://localhost:8080", "--backend-url"),
    max_samples: int = typer.Option(100, "--max-samples"),
    device: str = typer.Option(None, "--device"),
) -> None:
    """Reconstruction quality: predict c, re-render, compare to original."""
    reconstruct_fn(
        data_root=data_root,
        checkpoint=checkpoint,
        out_dir=out_dir,
        backend_url=backend_url,
        max_samples=max_samples,
        device_override=device,
    )


@app.command("train-v2")
def train_v2_cmd(
    data_root: Path = typer.Option(..., "--data-root"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(64, "--batch-size"),
    lr: float = typer.Option(1e-3, "--lr"),
    weight_decay: float = typer.Option(1e-4, "--weight-decay"),
    cls_weight: float = typer.Option(1.0, "--cls-weight"),
    c_weight: float = typer.Option(0.5, "--c-weight"),
    exp_weight: float = typer.Option(0.5, "--exp-weight"),
    vp_weight: float = typer.Option(
        0.25, "--vp-weight",
        help="Weight on the viewport MVE term. Always-on (no mask). Default 0.25 keeps "
             "viewport NLL from dominating; raise to >1 for reconstruction-heavy runs.",
    ),
    contrastive_weight: float = typer.Option(
        0.0, "--contrastive-weight",
        help="Supervised-contrastive weight on the embedding head. 0 disables (default). "
             "Typical: 0.1–1.0. The MVE / classifier heads are unaffected.",
    ),
    contrastive_temperature: float = typer.Option(
        0.1, "--contrastive-temperature",
        help="Softmax temperature for the SupCon loss. Lower = sharper.",
    ),
    base_channels: int = typer.Option(32, "--base-channels"),
    patience: int = typer.Option(8, "--patience"),
    device: str = typer.Option(None, "--device"),
    no_aug: bool = typer.Option(False, "--no-aug"),
    seed: int = typer.Option(42, "--seed"),
    eval_splits: list[str] = typer.Option(
        None,
        "--eval-split",
        help="Splits to evaluate the best checkpoint on at end of training. Repeat for several. "
             "Default: test, near_ood, hard_overlap.",
    ),
    from_best: Path = typer.Option(
        None,
        "--from-best",
        help="Load hyperparameters from a previous tune-v2 run's best_params_v2.json.",
    ),
) -> None:
    """Multi-task training (family + Julia c MVE + Multibrot exp MVE).

    Targets the Slice 2 dataset format (single labels.parquet with embedded
    `split` column). The legacy `train` command stays for back-compat with
    existing checkpoints.
    """
    if from_best is not None:
        params = load_best_params_v2(from_best)
        lr = float(params.get("learning_rate", lr))
        weight_decay = float(params.get("weight_decay", weight_decay))
        batch_size = int(params.get("batch_size", batch_size))
        c_weight = float(params.get("c_weight", c_weight))
        exp_weight = float(params.get("exp_weight", exp_weight))
        vp_weight = float(params.get("vp_weight", vp_weight))
        base_channels = int(params.get("base_channels", base_channels))
        if "enable_augmentation" in params:
            no_aug = not bool(params["enable_augmentation"])

    cfg = TrainingConfigV2(
        data_root=data_root,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        cls_weight=cls_weight,
        c_weight=c_weight,
        exp_weight=exp_weight,
        vp_weight=vp_weight,
        contrastive_weight=contrastive_weight,
        contrastive_temperature=contrastive_temperature,
        base_channels=base_channels,
        patience=patience,
        device_override=device,
        enable_augmentation=not no_aug,
        seed=seed,
        eval_splits=tuple(eval_splits) if eval_splits else ("test", "near_ood", "hard_overlap"),
    )
    run_training_v2(cfg)


@app.command("tune-v2")
def tune_v2_cmd(
    data_root: Path = typer.Option(..., "--data-root"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    n_trials: int = typer.Option(20, "--n-trials"),
    epochs_per_trial: int = typer.Option(10, "--epochs-per-trial"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option(None, "--device"),
) -> None:
    """Optuna search for v2 hyperparameters.

    Search space: lr, weight_decay, batch_size, c_weight (Julia NLL),
    exp_weight (Multibrot NLL), base_channels, augmentation. Objective
    minimised is val cls_loss — the same metric `train-v2` uses for
    best.pt selection.

    Pass the produced ``best_params_v2.json`` to
    ``train-v2 --from-best <out_dir>`` to train at full epoch budget on
    the chosen hyperparameters.
    """
    tcfg = TuneConfigV2(
        data_root=data_root,
        out_dir=out_dir,
        n_trials=n_trials,
        epochs_per_trial=epochs_per_trial,
        seed=seed,
        device_override=device,
    )
    run_study_v2(tcfg)


@app.command("eval-v2")
def eval_v2_cmd(
    data_root: Path = typer.Option(..., "--data-root"),
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    out_dir: Path = typer.Option(None, "--out-dir"),
    split: list[str] = typer.Option(
        None, "--split",
        help="Which splits to evaluate. Repeat. Default: test near_ood hard_overlap.",
    ),
    device: str = typer.Option(None, "--device"),
    base_channels: int = typer.Option(None, "--base-channels", help="Override checkpoint metadata if missing."),
    refine_iters: int = typer.Option(20, "--refine-iters"),
    refine_max_per_family: int = typer.Option(50, "--refine-max-per-family"),
) -> None:
    """Multi-task eval with hybrid reconstruction comparison."""
    splits = tuple(split) if split else ("test", "near_ood", "hard_overlap")
    evaluate_v2_fn(
        data_root=data_root,
        checkpoint=checkpoint,
        out_dir=out_dir,
        splits=splits,
        base_channels=base_channels,
        refine_n_iters=refine_iters,
        refine_max_samples_per_family=refine_max_per_family,
        device_override=device,
    )


@app.command("build-embeddings")
def build_embeddings_cmd(
    data_root: Path = typer.Option(..., "--data-root"),
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    out: Path = typer.Option(None, "--out", help="Output .npz path; default <data_root>/embeddings.npz."),
    split: list[str] = typer.Option(
        None, "--split",
        help="Splits to embed. Repeat. Default: train val test.",
    ),
    batch_size: int = typer.Option(64, "--batch-size"),
    device: str = typer.Option(None, "--device"),
    base_channels: int = typer.Option(None, "--base-channels"),
    embedding_dim: int = typer.Option(None, "--embedding-dim"),
) -> None:
    """Build the retrieval embedding bank from a trained v2 checkpoint.

    Embeds every requested split and writes a single ``embeddings.npz`` with
    aligned ``ids``, ``families``, ``splits``, and unit-norm embeddings.
    """
    splits = tuple(split) if split else ("train", "val", "test")
    embeddings_module.build_embedding_bank(
        data_root=data_root,
        checkpoint=checkpoint,
        splits=splits,
        out_path=out,
        base_channels=base_channels,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        device_override=device,
    )


@app.command()
def retrieve(
    bank: Path = typer.Option(..., "--bank", help="embeddings.npz built by build-embeddings."),
    image: Path = typer.Option(..., "--image", help="Query image path."),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Same checkpoint used to build the bank."),
    data_root: Path = typer.Option(None, "--data-root", help="Optional; if set, prints recipe rows for hits."),
    k: int = typer.Option(10, "--k"),
    family_filter: str = typer.Option(None, "--family", help="Restrict candidates to one family."),
    device: str = typer.Option(None, "--device"),
    base_channels: int = typer.Option(None, "--base-channels"),
    embedding_dim: int = typer.Option(None, "--embedding-dim"),
) -> None:
    """Query the embedding bank with an image and print top-K nearest neighbours."""
    console = Console()
    loaded_bank = embeddings_module.EmbeddingBank.load(bank)
    query = embeddings_module.encode_image(
        image,
        checkpoint=checkpoint,
        base_channels=base_channels,
        embedding_dim=embedding_dim,
        device_override=device,
    )
    hits = embeddings_module.retrieve_topk(
        loaded_bank, query, k=k, family_filter=family_filter,
    )
    if not hits:
        console.print("[yellow]no hits — empty bank or filter excluded everything")
        return

    from rich.table import Table
    table = Table(title=f"top-{len(hits)} for {image.name}")
    table.add_column("rank")
    table.add_column("id")
    table.add_column("family")
    table.add_column("split")
    table.add_column("cosine sim")
    for hit in hits:
        table.add_row(
            str(hit.rank),
            hit.id,
            hit.family,
            hit.split,
            f"{hit.similarity:.4f}",
        )
    console.print(table)

    if data_root is not None:
        rows = embeddings_module.lookup_recipe_rows(data_root, [h.id for h in hits])
        cols = [c for c in ("id", "family", "c_re", "c_im", "exponent",
                            "max_iter", "vp_x_min", "vp_x_max", "vp_y_min", "vp_y_max")
                if c in rows.columns]
        console.print(rows[cols].to_string(index=False))


@app.command()
def serve(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to best.pt."),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(9000, "--port"),
    device: str = typer.Option(None, "--device", help="cpu/mps/cuda; default = auto-pick."),
) -> None:
    """Run the FastAPI inference server on PORT, loading CHECKPOINT once at startup."""
    serve_module.run(checkpoint=checkpoint, host=host, port=port, device_override=device)


if __name__ == "__main__":
    app()
