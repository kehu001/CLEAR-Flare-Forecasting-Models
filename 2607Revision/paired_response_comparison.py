"""Paired SCI-versus-OPR performance comparison plots.

The input is a tidy table with one row per bootstrap result.  By default the
required columns are::

    response, period, window, task, predictor, model, bootstrap, metric, value

``response`` must identify the science-quality (``sci``) and operational
(``opr``) results.  Pairing is performed before summarising, so bootstrap 7 for
SCI is compared only with bootstrap 7 for OPR for the same period, window,
task, predictor, model, and metric.

Example
-------
python paired_response_comparison.py results.csv --predictor dft --model lstm

This writes ``figure/paired_response_comparison_dft_lstm.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_PERIOD_ORDER = ("2020-2021", "2022", "2023-2024", "2020-2024")
DEFAULT_WINDOW_ORDER = (0, 6, 12, 24)
DEFAULT_METRIC_ORDER = ("TSS", "HSS", "F1", "POD", "FAR", "ACC")
DEFAULT_PERIOD_LABELS = {
    "2020-01-01 to 2022-01-01": "2020-2021",
    "2022-01-01 to 2023-01-01": "2022",
    "2023-01-01 to 2025-01-01": "2023-2024",
    "2020-01-01 to 2025-01-01": "2020-2024",
}


def read_results(path: str | Path) -> pd.DataFrame:
    """Read a CSV, TSV, or Parquet table."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def wide_metrics_to_long(
    results: pd.DataFrame,
    *,
    metric_columns: Sequence[str] = DEFAULT_METRIC_ORDER,
    value_name: str = "value",
) -> pd.DataFrame:
    """Convert a table with TSS/HSS/etc. columns to the required long form."""
    present = [column for column in metric_columns if column in results.columns]
    if not present:
        raise ValueError(f"None of the metric columns are present: {list(metric_columns)}")
    identifiers = [column for column in results.columns if column not in present]
    return results.melt(
        id_vars=identifiers,
        value_vars=present,
        var_name="metric",
        value_name=value_name,
    )


def read_bootstrap_metric_csv(
    path: str | Path,
    *,
    response: str,
    window: int,
    task: str,
    predictor: str,
    model: str,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
) -> pd.DataFrame:
    """Read the repository's ``Metric,Period,Sample_1,...`` CSV layout.

    The returned frame is already compatible with
    :func:`plot_paired_response_comparison`.
    """
    table = pd.read_csv(path)
    required = {"Metric", "Period"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    sample_columns = [column for column in table if str(column).startswith("Sample_")]
    if not sample_columns:
        raise ValueError(f"{path} contains no Sample_* bootstrap columns")

    long = table.melt(
        id_vars=["Metric", "Period"],
        value_vars=sample_columns,
        var_name="bootstrap",
        value_name="value",
    ).rename(columns={"Metric": "metric", "Period": "period"})
    long["bootstrap"] = long["bootstrap"].str.extract(r"(\d+)$", expand=False).astype(int) - 1
    if period_labels:
        long["period"] = long["period"].replace(dict(period_labels))
    long["response"] = response
    long["window"] = int(window)
    long["task"] = task
    long["predictor"] = predictor
    long["model"] = model
    return long[
        ["response", "period", "window", "task", "predictor", "model",
         "bootstrap", "metric", "value"]
    ]


def load_response_metric_files(
    science_files: Mapping[int, str | Path],
    operational_files: Mapping[int, str | Path],
    *,
    task: str,
    predictor: str,
    model: str,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
) -> pd.DataFrame:
    """Load matched SCI/OPR metric CSVs keyed by forecast window."""
    if set(science_files) != set(operational_files):
        raise ValueError(
            "SCI and OPR file mappings must contain the same windows; "
            f"got SCI={sorted(science_files)} and OPR={sorted(operational_files)}"
        )
    frames = []
    for window in sorted(science_files):
        frames.append(
            read_bootstrap_metric_csv(
                science_files[window], response="sci", window=window, task=task,
                predictor=predictor, model=model, period_labels=period_labels,
            )
        )
        frames.append(
            read_bootstrap_metric_csv(
                operational_files[window], response="opr", window=window, task=task,
                predictor=predictor, model=model, period_labels=period_labels,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _normalise_window(value: object) -> int:
    text = str(value).strip().lower().replace("hours", "").replace("hour", "")
    text = text.replace("hrs", "").replace("hr", "").replace("h", "")
    if text in {"nowcast", "nowcasting", "lead0"}:
        return 0
    if text.startswith("lead"):
        text = text[4:]
    return int(float(text))


def _ordered_present(values: Iterable[object], preferred: Sequence[object]) -> list:
    values = list(pd.unique(pd.Series(list(values)).dropna()))
    ordered = [item for item in preferred if item in values]
    return ordered + sorted((item for item in values if item not in ordered), key=str)


def pair_response_results(
    results: pd.DataFrame,
    *,
    response_order: tuple[str, str] = ("sci", "opr"),
    response_col: str = "response",
    value_col: str = "value",
    pair_columns: Sequence[str] = (
        "period",
        "window",
        "task",
        "predictor",
        "model",
        "bootstrap",
        "metric",
    ),
    filters: Mapping[str, object] | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Return matched bootstrap differences (second response minus first).

    Parameters
    ----------
    strict:
        If True, reject duplicate keys and any SCI/OPR rows without a mate.
        This prevents silently treating independent or incomplete results as
        paired observations.
    """
    required = {response_col, value_col, *pair_columns}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = results.copy()
    if filters:
        for column, wanted in filters.items():
            if column not in results.columns:
                raise ValueError(f"Filter column {column!r} is absent")
            choices = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
            data = data[data[column].isin(choices)]
    data = data.loc[:, list(required)].copy()

    labels = tuple(str(x).strip().lower() for x in response_order)
    data[response_col] = data[response_col].astype(str).str.strip().str.lower()
    data = data[data[response_col].isin(labels)]
    data[value_col] = pd.to_numeric(data[value_col], errors="raise")
    if "window" in pair_columns:
        data["window"] = data["window"].map(_normalise_window)

    key = [*pair_columns, response_col]
    duplicate = data.duplicated(key, keep=False)
    if duplicate.any():
        example = data.loc[duplicate, key].head(5).to_dict("records")
        raise ValueError(f"Duplicate response rows for a pairing key; examples: {example}")

    wide = data.pivot(index=list(pair_columns), columns=response_col, values=value_col)
    counts = wide.notna().sum()
    missing_labels = [label for label in labels if label not in wide.columns]
    if missing_labels:
        raise ValueError(f"No observations found for response(s): {missing_labels}")

    incomplete = wide[list(labels)].isna().any(axis=1)
    if strict and incomplete.any():
        raise ValueError(
            f"Found {int(incomplete.sum())} unpaired bootstrap rows. "
            f"Matched counts are {counts.to_dict()}. Use strict=False only after review."
        )

    paired = wide.loc[~incomplete, list(labels)].reset_index()
    paired = paired.rename(columns={labels[0]: "sci_value", labels[1]: "opr_value"})
    paired["difference"] = paired["opr_value"] - paired["sci_value"]
    paired.attrs["difference_definition"] = f"{labels[1].upper()} - {labels[0].upper()}"
    return paired


def summarise_paired_differences(
    paired: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("period", "window", "task", "metric"),
) -> pd.DataFrame:
    """Summarise paired differences with median, 50%, and 95% intervals."""
    missing = set(group_columns).difference(paired.columns)
    if missing or "difference" not in paired:
        raise ValueError(f"Missing columns: {sorted(missing | ({'difference'} - set(paired)))}")

    summary = (
        paired.groupby(list(group_columns), observed=True, dropna=False)["difference"]
        .agg(
            median="median",
            low95=lambda x: x.quantile(0.025),
            low50=lambda x: x.quantile(0.25),
            high50=lambda x: x.quantile(0.75),
            high95=lambda x: x.quantile(0.975),
            n_pairs="size",
        )
        .reset_index()
    )
    return summary


def plot_paired_response_comparison(
    results: pd.DataFrame,
    output_path: str | Path = "figure/paired_response_comparison.pdf",
    *,
    predictor: str | None = None,
    model: str | None = None,
    tasks: Sequence[str] | None = None,
    metrics: Sequence[str] = DEFAULT_METRIC_ORDER,
    periods: Sequence[str] = DEFAULT_PERIOD_ORDER,
    windows: Sequence[int] = DEFAULT_WINDOW_ORDER,
    response_order: tuple[str, str] = ("sci", "opr"),
    strict: bool = True,
    title: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Create and save a faceted OPR-minus-SCI bootstrap comparison PDF.

    Predictor and model must each resolve to a single value; pooling them would
    make the uncertainty intervals scientifically ambiguous.  Tasks are shown
    in columns, metrics in rows, periods on the x-axis, and forecast windows as
    horizontally offset point/interval estimates.
    """
    data = results.copy()
    filters: dict[str, object] = {}
    if predictor is not None:
        filters["predictor"] = predictor
    if model is not None:
        filters["model"] = model
    if tasks is not None:
        filters["task"] = list(tasks)

    for dimension in ("predictor", "model"):
        selected = data
        if dimension in filters:
            selected = selected[selected[dimension].isin(
                filters[dimension] if isinstance(filters[dimension], list) else [filters[dimension]]
            )]
        unique = selected[dimension].dropna().unique() if dimension in selected else []
        if len(unique) != 1:
            raise ValueError(
                f"Select exactly one {dimension}; available values are {list(unique)}"
            )

    paired = pair_response_results(
        data, response_order=response_order, filters=filters, strict=strict
    )
    summary = summarise_paired_differences(paired)
    if summary.empty:
        raise ValueError("No paired observations remain after filtering")

    metric_order = _ordered_present(summary["metric"], metrics)
    task_order = _ordered_present(summary["task"], tasks or ("Task1", "Task2"))
    period_order = _ordered_present(summary["period"], periods)
    window_order = _ordered_present(summary["window"], windows)

    # Keep the overlapping full-period estimate visually separate.
    x_positions = np.arange(len(period_order), dtype=float)
    overall_index = next(
        (i for i, p in enumerate(period_order) if str(p).replace("–", "-") == "2020-2024"),
        None,
    )
    if overall_index is not None and overall_index > 0:
        x_positions[overall_index:] += 0.65

    nrows, ncols = len(metric_order), len(task_order)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.7 * ncols + 1.0, 2.25 * nrows + 1.2),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    colors = ("#1f5a92", "#c77c16", "#607d3b", "#9b4f78")
    markers = ("o", "s", "^", "D")
    offsets = np.linspace(-0.24, 0.24, len(window_order)) if len(window_order) > 1 else [0.0]

    for row, metric in enumerate(metric_order):
        for col, task in enumerate(task_order):
            ax = axes[row, col]
            panel = summary[(summary["metric"] == metric) & (summary["task"] == task)]
            ax.axhline(0, color="#303030", linewidth=0.9, zorder=0)
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.55, zorder=0)

            for j, window in enumerate(window_order):
                group = panel[panel["window"] == window].set_index("period")
                group = group.reindex(period_order)
                valid = group["median"].notna().to_numpy()
                if not valid.any():
                    continue
                x = x_positions[valid] + offsets[j]
                med = group.loc[group.index[valid], "median"].to_numpy(float)
                low95 = group.loc[group.index[valid], "low95"].to_numpy(float)
                high95 = group.loc[group.index[valid], "high95"].to_numpy(float)
                low50 = group.loc[group.index[valid], "low50"].to_numpy(float)
                high50 = group.loc[group.index[valid], "high50"].to_numpy(float)
                color = colors[j % len(colors)]
                ax.vlines(x, low95, high95, color=color, linewidth=1.0, zorder=2)
                ax.vlines(x, low50, high50, color=color, linewidth=3.4, zorder=3)
                ax.scatter(
                    x,
                    med,
                    s=28,
                    marker=markers[j % len(markers)],
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.25,
                    zorder=4,
                )

            if overall_index is not None and overall_index > 0:
                split = (x_positions[overall_index - 1] + x_positions[overall_index]) / 2
                ax.axvline(split, color="#a0a0a0", linewidth=0.75, linestyle=(0, (2, 2)))
            if row == 0:
                ax.set_title(str(task).replace("Task", "Task "), fontsize=10, pad=7)
            if col == 0:
                ax.set_ylabel(f"Delta {metric}\n(OPR - SCI)", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xticks(x_positions, period_order, rotation=0)
        ax.set_xlabel("Testing period", fontsize=9)

    handles = [
        Line2D(
            [0], [0], color=colors[j % len(colors)], marker=markers[j % len(markers)],
            markerfacecolor="white", linewidth=1.2, label="Nowcast" if w == 0 else f"{w} h"
        )
        for j, w in enumerate(window_order)
    ]
    default_title = "Paired response comparison"
    if predictor is not None or model is not None:
        context = ", ".join(x for x in (predictor, model) if x is not None)
        default_title += f" ({context})"
    # Keep the title and shared legend on separate lines in both screen and PDF output.
    fig.suptitle(title or default_title, fontsize=12, y=1.055)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": title or default_title,
        "Subject": "Bootstrap-paired OPR minus SCI performance differences",
        "Keywords": "solar flare forecasting, paired bootstrap, SCI, OPR",
    }
    with PdfPages(output_path, metadata=metadata) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    return fig, paired, summary


def _demo_results(seed: int = 2402) -> pd.DataFrame:
    """Deterministic synthetic data used only to preview/QA the layout."""
    rng = np.random.default_rng(seed)
    rows = []
    metrics = ("TSS", "HSS", "F1", "POD")
    for task_i, task in enumerate(("Task1", "Task2")):
        for period_i, period in enumerate(DEFAULT_PERIOD_ORDER):
            for window_i, window in enumerate(DEFAULT_WINDOW_ORDER):
                for bootstrap in range(30):
                    for metric_i, metric in enumerate(metrics):
                        base = 0.22 + 0.07 * metric_i + 0.035 * period_i - 0.012 * window_i
                        shared = rng.normal(0, 0.025)
                        delta = 0.018 + 0.007 * period_i - 0.005 * window_i + 0.004 * task_i
                        for response, shift in (("sci", 0.0), ("opr", delta)):
                            rows.append(
                                {
                                    "response": response,
                                    "period": period,
                                    "window": window,
                                    "task": task,
                                    "predictor": "dft",
                                    "model": "lstm",
                                    "bootstrap": bootstrap,
                                    "metric": metric,
                                    "value": base + shared + shift + rng.normal(0, 0.008),
                                }
                            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Long-form CSV, TSV, or Parquet results")
    parser.add_argument("--predictor", help="Predictor value to plot, e.g. dft or nrt")
    parser.add_argument("--model", help="Model value to plot, e.g. lstm or logistic")
    parser.add_argument("--task", action="append", dest="tasks", help="Task to include; repeatable")
    parser.add_argument("--metric", action="append", dest="metrics", help="Metric to include; repeatable")
    parser.add_argument("--output", help="Output PDF path")
    parser.add_argument("--allow-unpaired", action="store_true", help="Drop unmatched rows after review")
    parser.add_argument("--demo", action="store_true", help="Render a clearly synthetic layout preview")
    args = parser.parse_args()

    if args.demo:
        results = _demo_results()
        predictor, model = "dft", "lstm"
        output = args.output or "figure/paired_response_comparison_demo.pdf"
        title = "Paired response comparison - layout preview (synthetic data)"
    else:
        if not args.input:
            parser.error("input is required unless --demo is used")
        results = read_results(args.input)
        predictor, model = args.predictor, args.model
        suffix = "_".join(x for x in (predictor, model) if x) or "selected"
        output = args.output or f"figure/paired_response_comparison_{suffix}.pdf"
        title = None

    fig, paired, _ = plot_paired_response_comparison(
        results,
        output,
        predictor=predictor,
        model=model,
        tasks=args.tasks,
        metrics=args.metrics or DEFAULT_METRIC_ORDER,
        strict=not args.allow_unpaired,
        title=title,
    )
    plt.close(fig)
    print(f"Saved {output} using {len(paired):,} paired bootstrap observations")


if __name__ == "__main__":
    main()
