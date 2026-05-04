"""Quick EDA for a generated dataset.

Prints, in order:
  * dataset shape + family distribution
  * per-family numeric param distributions (median / IQR for compactness)
  * render time stats — useful to track Stage 10 perf regressions
  * split distribution (if splits.parquet is present)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


def inspect_dataset(root: Path, console: Console | None = None) -> None:
    console = console or Console()
    labels = pd.read_parquet(root / "labels.parquet")
    console.rule(f"[bold]{root.resolve()}")
    console.print(f"rows={len(labels):,}  unique families={labels['family'].nunique()}")

    _print_family_distribution(console, labels)
    _print_param_summary(console, labels)
    _print_perf_summary(console, labels)
    _maybe_print_splits(console, root, labels)


def _print_family_distribution(console: Console, labels: pd.DataFrame) -> None:
    table = Table(title="family distribution", show_header=True)
    table.add_column("family")
    table.add_column("count", justify="right")
    table.add_column("share", justify="right")
    counts = labels["family"].value_counts()
    total = int(counts.sum())
    for family, n in counts.items():
        table.add_row(str(family), f"{n}", f"{n / total:.1%}")
    console.print(table)


def _print_param_summary(console: Console, labels: pd.DataFrame) -> None:
    table = Table(title="numeric params (median ± IQR)", show_header=True)
    table.add_column("family")
    table.add_column("max_iter")
    table.add_column("c_re")
    table.add_column("c_im")
    table.add_column("exponent")
    table.add_column("vp_span_x")

    labels = labels.assign(vp_span_x=labels["vp_x_max"] - labels["vp_x_min"])
    for family, group in labels.groupby("family"):
        table.add_row(
            str(family),
            _fmt_iqr(group["max_iter"]),
            _fmt_iqr(group["c_re"]),
            _fmt_iqr(group["c_im"]),
            _fmt_iqr(group["exponent"]),
            _fmt_iqr(group["vp_span_x"]),
        )
    console.print(table)


def _print_perf_summary(console: Console, labels: pd.DataFrame) -> None:
    table = Table(title="render time per stage (ms, median ± IQR)", show_header=True)
    table.add_column("family")
    table.add_column("render_ms")
    table.add_column("colorize_ms")
    table.add_column("encode_ms")
    table.add_column("total_ms")
    for family, group in labels.groupby("family"):
        table.add_row(
            str(family),
            _fmt_iqr(group["perf_render_ms"]),
            _fmt_iqr(group["perf_colorize_ms"]),
            _fmt_iqr(group["perf_encode_ms"]),
            _fmt_iqr(group["perf_total_ms"]),
        )
    console.print(table)


def _maybe_print_splits(console: Console, root: Path, labels: pd.DataFrame) -> None:
    splits_path = root / "splits.parquet"
    if not splits_path.exists():
        console.print("[dim](no splits.parquet — run `fractalov-ml split` to create one)")
        return
    splits = pd.read_parquet(splits_path)
    merged = labels.merge(splits, on="id", how="inner")
    table = Table(title="split distribution", show_header=True)
    table.add_column("split")
    for fam in sorted(labels["family"].unique()):
        table.add_column(str(fam), justify="right")
    table.add_column("total", justify="right")
    for split_name in ("train", "val", "test"):
        sub = merged[merged["split"] == split_name]
        row = [split_name]
        for fam in sorted(labels["family"].unique()):
            row.append(str(int((sub["family"] == fam).sum())))
        row.append(str(len(sub)))
        table.add_row(*row)
    console.print(table)


def _fmt_iqr(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return "-"
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    if pd.api.types.is_integer_dtype(series) or series.dtype.kind in {"i", "u"}:
        return f"{int(median)} ({int(q1)}..{int(q3)})"
    return f"{median:.3f} ({q1:.3f}..{q3:.3f})"
