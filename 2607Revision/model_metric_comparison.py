"""Publication-ready paired metric-difference plots from ``results`` CSV files.

The repository stores each configuration in a file named
``{task}_raw_metrics_{model}-{response}-{predictor}-{window}.csv``.  Each
``Sample_*`` column is treated as a matched bootstrap replicate, so the figure
shows the distribution of *paired* differences (configuration 1 minus
configuration 2), not a difference of independently summarised estimates.

Example
-------
>>> from model_metric_comparison import plot_metric_difference_comparison
>>> fig, differences = plot_metric_difference_comparison(
...     response1="S", response2="S", predictor1="DEF", predictor2="NRT",
...     model1="LSTM", model2="LSTM", task="Task1",
...     output_path="figure/task1_def_vs_nrt.pdf",
... )

The left side shows paired differences (positive values favour configuration
1). The right side shows absolute median performance and 95% bootstrap
percentile intervals for both configurations and all forecast windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

# This module is normally run on a server or through an import, not from an
# interactive desktop session.  Selecting a non-GUI backend makes PDF/PNG
# exports reliable in both contexts.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


DEFAULT_WINDOWS = (0, 6, 12, 24)
DEFAULT_PERIOD_LABELS = {
    "2020-01-01 to 2022-01-01": "2020-2021",
    "2022-01-01 to 2025-01-01": "2022-2024",
    "2020-01-01 to 2025-01-01": "2020-2024",
}
DEFAULT_PERIOD_ORDER = ("2020-2021", "2022-2024")
# Okabe-Ito colours: colour-blind safe categorical window encoding.
WINDOW_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def _configuration_label(response: str, predictor: str, model: str) -> str:
    """Return a concise, human-readable configuration label."""
    return f"{model}-{response}-{predictor}"


def _result_file(
    results_dir: str | Path,
    *,
    task: str,
    model: str,
    response: str,
    predictor: str,
    window: int,
) -> Path:
    """Find one result CSV, allowing case-insensitive configuration inputs."""
    directory = Path(results_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {directory}")

    expected = f"{task}_raw_metrics_{model}-{response}-{predictor}-{window}.csv"
    direct = directory / expected
    if direct.is_file():
        return direct
    matches = [p for p in directory.glob("*.csv") if p.name.lower() == expected.lower()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find {expected!r} in {directory}. "
        "Check task, response, predictor, model, and forecast-window values."
    )


def read_bootstrap_metrics(
    results_dir: str | Path,
    *,
    task: str,
    response: str,
    predictor: str,
    model: str,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
) -> pd.DataFrame:
    """Read one configuration into tidy paired-bootstrap metric data.

    Returns columns ``period``, ``window``, ``metric``, ``bootstrap``, and
    ``value``.  A missing requested window is an error rather than silently
    producing a partial comparison.
    """
    frames: list[pd.DataFrame] = []
    for window in windows:
        path = _result_file(
            results_dir, task=task, model=model, response=response,
            predictor=predictor, window=int(window),
        )
        table = pd.read_csv(path)
        required = {"Metric", "Period"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        samples = [column for column in table.columns if str(column).startswith("Sample_")]
        if not samples:
            raise ValueError(f"{path} has no Sample_* bootstrap columns")
        tidy = table.melt(
            id_vars=["Metric", "Period"], value_vars=samples,
            var_name="bootstrap", value_name="value",
        ).rename(columns={"Metric": "metric", "Period": "period"})
        tidy["bootstrap"] = tidy["bootstrap"].str.extract(r"(\d+)$", expand=False)
        if tidy["bootstrap"].isna().any():
            raise ValueError(f"Could not identify a replicate number in {path}")
        tidy["bootstrap"] = tidy["bootstrap"].astype(int)
        tidy["value"] = pd.to_numeric(tidy["value"], errors="raise")
        tidy["window"] = int(window)
        if period_labels is not None:
            tidy["period"] = tidy["period"].replace(dict(period_labels))
        frames.append(tidy[["period", "window", "metric", "bootstrap", "value"]])
    return pd.concat(frames, ignore_index=True)


def paired_metric_differences(
    results_dir: str | Path = "results",
    *,
    response1: str,
    response2: str,
    predictor1: str,
    predictor2: str,
    model1: str,
    model2: str,
    task: str = "Task1",
    metrics: Sequence[str] = ("TSS", "F1"),
    windows: Sequence[int] = DEFAULT_WINDOWS,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
) -> pd.DataFrame:
    """Return matched bootstrap differences, computed as configuration 1 - 2.

    The function rejects unmatched metric/period/window/replicate keys.  This
    protects against comparing different test-period definitions or bootstrap
    samples that cannot be paired scientifically.
    """
    if not metrics:
        raise ValueError("Provide at least one metric")
    first = read_bootstrap_metrics(
        results_dir, task=task, response=response1, predictor=predictor1,
        model=model1, windows=windows, period_labels=period_labels,
    )
    second = read_bootstrap_metrics(
        results_dir, task=task, response=response2, predictor=predictor2,
        model=model2, windows=windows, period_labels=period_labels,
    )
    wanted = list(metrics)
    first = first[first["metric"].isin(wanted)].copy()
    second = second[second["metric"].isin(wanted)].copy()
    for name, frame in (("configuration 1", first), ("configuration 2", second)):
        absent = sorted(set(wanted).difference(frame["metric"].unique()))
        if absent:
            raise ValueError(f"{name} has no values for metric(s): {absent}")

    keys = ["period", "window", "metric", "bootstrap"]
    for name, frame in (("configuration 1", first), ("configuration 2", second)):
        duplicate = frame.duplicated(keys, keep=False)
        if duplicate.any():
            examples = frame.loc[duplicate, keys].head(3).to_dict("records")
            raise ValueError(f"Duplicate pairing keys in {name}: {examples}")
    paired = first.merge(second, on=keys, how="outer", validate="one_to_one",
                         suffixes=("_1", "_2"), indicator=True)
    if not (paired["_merge"] == "both").all():
        unmatched = paired.loc[paired["_merge"] != "both", keys + ["_merge"]].head(8)
        raise ValueError(
            "The selected configurations do not share identical testing "
            "periods/metrics/bootstrap replicates. Examples: "
            f"{unmatched.to_dict('records')}"
        )
    paired = paired.drop(columns="_merge")
    paired["difference"] = paired["value_1"] - paired["value_2"]
    paired["configuration_1"] = _configuration_label(response1, predictor1, model1)
    paired["configuration_2"] = _configuration_label(response2, predictor2, model2)
    return paired


def _present_order(values: Sequence[object], preferred: Sequence[object]) -> list[object]:
    present = list(pd.unique(pd.Series(values).dropna()))
    return [x for x in preferred if x in present] + sorted(
        (x for x in present if x not in preferred), key=str
    )


def plot_metric_difference_comparison(
    results_dir: str | Path = "results",
    output_path: str | Path | None = "figure/model_metric_difference.pdf",
    *,
    response1: str = "S",
    response2: str = "S",
    predictor1: str = "DEF",
    predictor2: str = "NRT",
    model1: str = "LSTM",
    model2: str = "LSTM",
    metric1: str = "TSS",
    metric2: str = "F1",
    task: str = "Task1",
    windows: Sequence[int] = DEFAULT_WINDOWS,
    periods: Sequence[str] = DEFAULT_PERIOD_ORDER,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
    display: str = "box",
    dpi: int = 300,
    title: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot paired differences with compact absolute-performance panels.

    Parameters ``response1`` through ``model2`` define the two configurations.
    Each period contains four offset distributions, one for every forecast
    window. In the right panels, forecast windows are encoded by colour and
    configurations by marker style; a short connector joins the two
    configurations within each period, with small horizontal offsets to keep
    estimates legible. ``display='box'`` (the default) makes the left side
    boxplots of the bootstrap differences; ``display='median_ci'`` uses median
    point/interval summaries there instead.
    """
    if display not in {"box", "median_ci"}:
        raise ValueError("display must be 'box' or 'median_ci'")
    metrics = (metric1, metric2)
    if len(set(metrics)) != 2:
        raise ValueError("metric1 and metric2 must be different for a 2 x 1 figure")
    differences = paired_metric_differences(
        results_dir, response1=response1, response2=response2,
        predictor1=predictor1, predictor2=predictor2, model1=model1,
        model2=model2, task=task, metrics=metrics, windows=windows,
        period_labels=period_labels,
    )
    # Do not append unrequested (overlapping) periods. In particular, the
    # default excludes the 2020-2024 aggregate because it reuses observations
    # already represented by the two disjoint testing periods.
    period_order = [period for period in periods if period in set(differences["period"])]
    if not period_order:
        raise ValueError(f"None of the requested periods are present: {list(periods)}")
    differences = differences[differences["period"].isin(period_order)].copy()
    window_order = _present_order(differences["window"], windows)
    metric_order = list(metrics)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    # Left: paired-difference distributions. Right: all absolute values in one
    # compact panel per metric. Colour distinguishes forecast window; marker
    # and line style distinguish configuration, avoiding eight boxes/period.
    fig = plt.figure(figsize=(9.2, 5.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.0))
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.86))
    difference_axes = [fig.add_subplot(grid[0, 0])]
    difference_axes.append(fig.add_subplot(grid[1, 0], sharex=difference_axes[0]))
    absolute_axes = [fig.add_subplot(grid[0, 1])]
    absolute_axes.append(fig.add_subplot(grid[1, 1], sharex=absolute_axes[0]))
    config1 = _configuration_label(response1, predictor1, model1)
    config2 = _configuration_label(response2, predictor2, model2)
    x_centres = np.arange(len(period_order), dtype=float)
    offsets = np.linspace(-0.29, 0.29, len(window_order)) if len(window_order) > 1 else np.array([0.0])
    width = min(0.17, 0.72 / max(len(window_order), 1))

    for panel_index, (ax, metric) in enumerate(zip(difference_axes, metric_order)):
        panel = differences[differences["metric"] == metric]
        ax.axhline(0, color="#333333", linewidth=0.8, zorder=0)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
        for j, window in enumerate(window_order):
            data = [
                panel.loc[(panel["period"] == period) & (panel["window"] == window), "difference"].to_numpy()
                for period in period_order
            ]
            positions = x_centres + offsets[j]
            color = WINDOW_COLORS[j % len(WINDOW_COLORS)]
            if display == "box":
                boxes = ax.boxplot(
                    data, positions=positions, widths=width, patch_artist=True,
                    manage_ticks=False, showfliers=False, whis=1.5,
                    medianprops={"color": "#202020", "linewidth": 1.15},
                    whiskerprops={"color": "#4A4A4A", "linewidth": 0.75},
                    capprops={"color": "#4A4A4A", "linewidth": 0.75},
                    boxprops={"edgecolor": "#3A3A3A", "linewidth": 0.75},
                )
                for box in boxes["boxes"]:
                    box.set_facecolor(color)
                    box.set_alpha(0.78)
            else:
                for period_index, values in enumerate(data):
                    if not values.size:
                        continue
                    median = np.median(values)
                    low, high = np.quantile(values, [0.025, 0.975])
                    ax.errorbar(positions[period_index], median,
                                yerr=[[median - low], [high - median]],
                                fmt="o", markersize=4.3, capsize=2.5,
                                color=color, markeredgecolor="#303030",
                                markeredgewidth=0.45, elinewidth=1.0, zorder=4)
        ax.set_ylabel(f"{metric} difference\n(config. 1 - config. 2)")
        ax.set_title(f"{chr(97 + panel_index)}  {metric}: {config1} - {config2}", loc="left",
                     fontweight="bold", pad=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        ax.margins(x=0.06, y=0.11)

    difference_axes[-1].set_xticks(x_centres, period_order)
    difference_axes[-1].set_xlabel("Testing period")
    absolute_offsets = np.linspace(-0.27, 0.27, len(window_order))
    response_offsets = (-0.034, 0.034)
    for panel_index, (metric, ax) in enumerate(zip(metric_order, absolute_axes)):
        panel = differences[differences["metric"] == metric]
        for j, window in enumerate(window_order):
            summaries = []
            for response_index, (values, marker, label) in enumerate(
                (("value_1", "o", config1), ("value_2", "s", config2))
            ):
                summary = (
                    panel.loc[panel["window"] == window]
                    .groupby("period", observed=True)[values]
                    .agg(median="median", low=lambda x: x.quantile(0.025),
                         high=lambda x: x.quantile(0.975))
                    .reindex(period_order)
                )
                summaries.append(summary)
                x = np.arange(len(period_order)) + absolute_offsets[j] + response_offsets[response_index]
                median = summary["median"].to_numpy(float)
                ax.errorbar(
                    x, median,
                    yerr=np.vstack((median - summary["low"].to_numpy(float),
                                    summary["high"].to_numpy(float) - median)),
                    color=WINDOW_COLORS[j % len(WINDOW_COLORS)], linestyle="none",
                    marker=marker, markersize=3.7,
                    linewidth=1.0, elinewidth=0.8, capsize=1.8,
                    markeredgecolor="#303030", markeredgewidth=0.35, zorder=3,
                    label=label,
                )
            # The connector encodes the configuration comparison within one
            # testing period.  No line crosses from one period to another.
            for period_index in range(len(period_order)):
                pair = [summary["median"].iloc[period_index] for summary in summaries]
                if np.isfinite(pair).all():
                    x_pair = [
                        period_index + absolute_offsets[j] + response_offsets[0],
                        period_index + absolute_offsets[j] + response_offsets[1],
                    ]
                    ax.plot(x_pair, pair, color=WINDOW_COLORS[j % len(WINDOW_COLORS)],
                            linewidth=0.9, zorder=2)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.set_ylabel(f"{metric} score")
        ax.set_title(f"{chr(99 + panel_index)}  {metric}: absolute performance", loc="left",
                     fontweight="bold", pad=5)
        ax.margins(x=0.13, y=0.11)
        if panel_index == 1:
            ax.set_xticks(np.arange(len(period_order)), period_order)
            ax.set_xlabel("Testing period")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    window_handles = [
        Patch(facecolor=WINDOW_COLORS[j % len(WINDOW_COLORS)], edgecolor="#3A3A3A",
              alpha=0.78, label="Nowcast" if window == 0 else f"{window} h")
        for j, window in enumerate(window_order)
    ]
    response_handles = [
        Line2D([], [], color="#404040", marker="o", linestyle="none", markersize=4,
               linewidth=1.0, label=config1),
        Line2D([], [], color="#404040", marker="s", linestyle="none", markersize=4,
               linewidth=1.0, label=config2),
    ]
    fig.legend(handles=window_handles, loc="upper center", ncol=len(window_order),
               frameon=False, fontsize=7.2, bbox_to_anchor=(0.27, 0.955),
               columnspacing=1.2, handlelength=1.5)
    fig.legend(handles=response_handles, loc="upper center", ncol=2, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.76, 0.955), columnspacing=1.2,
               handlelength=1.8)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, differences
