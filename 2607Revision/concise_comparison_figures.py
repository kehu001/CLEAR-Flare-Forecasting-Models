"""Compact, publication-oriented paired-effect figures for flare forecasts.

The existing figures in :mod:`model_metric_comparison` document every
configuration separately.  This module is for the manuscript-level summary:
one four-panel figure per comparison factor, with Task 1/Task 2 as rows and
the two independent test periods as columns.

The primary use is::

    from concise_comparison_figures import plot_response_list_summary

    fig, summary = plot_response_list_summary(
        results_dir="results",
        output_path="figure/concise/response_list_tss.pdf",
    )

Positive effects always favour the first named data source/model.  Error bars
are paired 95% bootstrap percentile intervals, not hypothesis-test confidence
intervals.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from model_metric_comparison import (
    DEFAULT_PERIOD_ORDER,
    DEFAULT_WINDOWS,
    paired_metric_differences,
)


Factor = Literal["response", "predictor", "model"]

_FACTOR_SPECS = {
    "response": {
        "varying": ("model", "predictor"),
        "left": {"response": "S"},
        "right": {"response": "O"},
        "effect_label": "Science-quality - operational response",
        "short_left": "Science-quality",
        "short_right": "Operational",
    },
    "predictor": {
        "varying": ("model", "response"),
        "left": {"predictor": "DEF"},
        "right": {"predictor": "NRT"},
        "effect_label": "Definitive - near-real-time predictor",
        "short_left": "DEF",
        "short_right": "NRT",
    },
    "model": {
        "varying": ("response", "predictor"),
        "left": {"model": "LSTM"},
        "right": {"model": "Logreg"},
        "effect_label": "LSTM - logistic regression",
        "short_left": "LSTM",
        "short_right": "Logistic regression",
    },
}

_LEVELS = {
    "model": ("LSTM", "Logreg"),
    "predictor": ("DEF", "NRT"),
    "response": ("S", "O"),
}
_DISPLAY = {"LSTM": "LSTM", "Logreg": "Logistic", "DEF": "DEF", "NRT": "NRT", "S": "S", "O": "O"}
# Purple and teal avoid the blue/orange encoding already used for the
# science-quality and operational response lists elsewhere in the manuscript.
_COLORS = ("#7A5195", "#009E73")
_MARKERS = ("o", "s")


def summarise_factor_effects(
    factor: Factor,
    *,
    results_dir: str | Path = "results",
    metric: str = "TSS",
    tasks: Sequence[str] = ("Task1", "Task2"),
    periods: Sequence[str] = DEFAULT_PERIOD_ORDER,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Return paired-bootstrap effect summaries for one comparison factor.

    ``factor`` selects the signed comparison: ``response`` is S - O,
    ``predictor`` is DEF - NRT, and ``model`` is LSTM - logistic regression.
    The two non-focal settings are retained as line-defining configurations.
    The returned table contains one row per task, period, lead time, and
    line-defining configuration, with the paired median and 95% percentile
    interval calculated from the matched bootstrap replicates.
    """
    if factor not in _FACTOR_SPECS:
        raise ValueError(f"factor must be one of {tuple(_FACTOR_SPECS)}, got {factor!r}")
    if not tasks or not periods or not windows:
        raise ValueError("tasks, periods, and windows must all be non-empty")

    spec = _FACTOR_SPECS[factor]
    varying = spec["varying"]
    frames: list[pd.DataFrame] = []
    for task in tasks:
        for values in product(*(_LEVELS[name] for name in varying)):
            fixed = dict(zip(varying, values, strict=True))
            first = {**fixed, **spec["left"]}
            second = {**fixed, **spec["right"]}
            paired = paired_metric_differences(
                results_dir,
                task=task,
                response1=first["response"], response2=second["response"],
                predictor1=first["predictor"], predictor2=second["predictor"],
                model1=first["model"], model2=second["model"],
                metrics=(metric,), windows=windows,
            )
            paired = paired[paired["period"].isin(periods)].copy()
            # ``paired_metric_differences`` validates and returns the selected
            # task, but does not retain its name as a column.  Add it here so
            # that the manuscript figure can facet Task 1 and Task 2.
            paired["task"] = task
            for name, value in fixed.items():
                paired[name] = value
            frames.append(paired)

    paired = pd.concat(frames, ignore_index=True)
    group_columns = ["task", "period", "window", *varying]
    summary = (
        paired.groupby(group_columns, observed=True, as_index=False)["difference"]
        .agg(
            median="median",
            low95=lambda values: values.quantile(0.025),
            high95=lambda values: values.quantile(0.975),
            n_bootstrap="size",
        )
    )
    expected = len(tasks) * len(periods) * len(windows) * np.prod([len(_LEVELS[name]) for name in varying])
    if len(summary) != expected:
        raise ValueError(f"Expected {expected} paired summaries but constructed {len(summary)}")
    return summary


def plot_factor_effect_summary(
    factor: Factor,
    *,
    results_dir: str | Path = "results",
    output_path: str | Path | None = None,
    metric: str = "TSS",
    tasks: Sequence[str] = ("Task1", "Task2"),
    periods: Sequence[str] = DEFAULT_PERIOD_ORDER,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    title: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot a compact 2 x 2 paired-effect figure for a manuscript.

    Rows represent forecasting tasks and columns represent independent test
    periods.  Each panel shows four fixed configurations as lead-time curves:
    colour encodes the first non-focal setting and marker shape encodes the
    second.  The common y-axis is a signed paired difference, so the zero line
    provides a direct visual test of the direction of the effect.

    The default response figure is suitable for the main text.  Predictor and
    model figures use the same design and are suitable as supplementary
    figures, preventing selective presentation of individual comparisons.
    """
    if len(tasks) != 2 or len(periods) != 2:
        raise ValueError("The compact layout requires exactly two tasks and two periods")
    summary = summarise_factor_effects(
        factor, results_dir=results_dir, metric=metric, tasks=tasks,
        periods=periods, windows=windows,
    )
    spec = _FACTOR_SPECS[factor]
    varying = spec["varying"]
    line_combinations = list(product(*(_LEVELS[name] for name in varying)))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8.5, "axes.titlesize": 9.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15), sharex=True, sharey=True,
                             constrained_layout=True)
    # Reserve a dedicated legend band.  Without this margin, the legends can
    # overlap the upper-row panel titles after vector export.
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.81))

    max_abs = float(np.abs(summary[["low95", "high95"]].to_numpy()).max())
    y_limit = max(0.05, max_abs * 1.14)
    lead_positions = np.arange(len(windows), dtype=float)
    # Four configurations occur at each lead-time category.  Small symmetric
    # offsets keep their markers and percentile intervals readable without
    # changing the categorical meaning of the x-axis.
    line_offsets = np.linspace(-0.18, 0.18, len(line_combinations))
    panel_index = 0
    for row, task in enumerate(tasks):
        for column, period in enumerate(periods):
            ax = axes[row, column]
            panel = summary[(summary["task"] == task) & (summary["period"] == period)]
            ax.axhline(0, color="#3A3A3A", linewidth=0.85, zorder=0)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
            ax.set_axisbelow(True)
            for line_index, values in enumerate(line_combinations):
                selectors = dict(zip(varying, values, strict=True))
                line = panel.copy()
                for name, value in selectors.items():
                    line = line[line[name] == value]
                line = line.set_index("window").reindex(windows)
                if line[["median", "low95", "high95"]].isna().any().any():
                    raise ValueError(f"Missing result(s) for {task}, {period}, {selectors}")
                color = _COLORS[line_index // len(_MARKERS)]
                marker = _MARKERS[line_index % len(_MARKERS)]
                median = line["median"].to_numpy(float)
                low = line["low95"].to_numpy(float)
                high = line["high95"].to_numpy(float)
                ax.errorbar(
                    lead_positions + line_offsets[line_index], median,
                    yerr=np.vstack((median - low, high - median)),
                    fmt=marker, linestyle="none", color=color, markersize=4.3,
                    markeredgecolor="#333333", markeredgewidth=0.4,
                    elinewidth=0.85, capsize=2.2,
                    zorder=3,
                )
            ax.set_ylim(-y_limit, y_limit)
            ax.set_xlim(-0.48, len(windows) - 0.52)
            ax.set_xticks(lead_positions, ["Nowcast" if window == 0 else f"{window} h" for window in windows])
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=7.5)
            ax.set_title(f"{chr(97 + panel_index)}  {task}, {period}", loc="left", fontweight="bold", pad=9)
            panel_index += 1

    for ax in axes[:, 0]:
        ax.set_ylabel(f"Paired {metric} difference\n({spec['short_left']} - {spec['short_right']})")
    for ax in axes[-1, :]:
        ax.set_xlabel("Forecast lead time")
    if title:
        fig.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)

    # The legend separates colour and marker semantics, so the four curves do
    # not require four long, repetitive labels.
    first, second = varying
    colour_handles = [
        Line2D([], [], color=_COLORS[index], marker="o", linestyle="none", markersize=4,
               label=_DISPLAY[value])
        for index, value in enumerate(_LEVELS[first])
    ]
    marker_handles = [
        Line2D([], [], color="#4A4A4A", marker=_MARKERS[index], linestyle="none", markersize=4.5,
               label=_DISPLAY[value])
        for index, value in enumerate(_LEVELS[second])
    ]
    legend_one = fig.legend(colour_handles, [_DISPLAY[value] for value in _LEVELS[first]],
                            title=first.capitalize(), loc="upper center", ncol=2,
                            frameon=False, fontsize=7.2, title_fontsize=7.2,
                            bbox_to_anchor=(0.38, 0.90), columnspacing=1.2, handlelength=1.2)
    fig.add_artist(legend_one)
    fig.legend(marker_handles, [_DISPLAY[value] for value in _LEVELS[second]],
               title=second.capitalize(), loc="upper center", ncol=2,
               frameon=False, fontsize=7.2, title_fontsize=7.2,
               bbox_to_anchor=(0.77, 0.90), columnspacing=1.2, handlelength=1.2)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    return fig, summary


def plot_response_list_summary(**kwargs: object) -> tuple[plt.Figure, pd.DataFrame]:
    """Create the recommended main-text figure: paired TSS effect of S vs O."""
    return plot_factor_effect_summary("response", **kwargs)


def write_effect_summary_suite(
    *,
    results_dir: str | Path = "results",
    output_directory: str | Path = "figure/concise",
    metric: str = "TSS",
) -> dict[str, tuple[plt.Figure, pd.DataFrame]]:
    """Write consistent response, predictor, and model summary figures.

    The response figure is suitable for the main manuscript; predictor and
    model figures provide concise, non-selective supplementary evidence.
    Output filenames are stable and use PDF vector graphics.
    """
    output_directory = Path(output_directory)
    outputs: dict[str, tuple[plt.Figure, pd.DataFrame]] = {}
    for factor in ("response", "predictor", "model"):
        outputs[factor] = plot_factor_effect_summary(
            factor, results_dir=results_dir, metric=metric,
            output_path=output_directory / f"{factor}_effect_{metric.lower()}.pdf",
        )
    return outputs
