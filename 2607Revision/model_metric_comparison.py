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
# Academic-figure-skill palette.  Keep these definitions centralized so that
# future figures in this module use the same publication-safe colours.
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7",
    "#FDDBC7", "#F4A582", "#D6604D", "#B2182B",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

# Okabe-Ito colours retained for the older window-encoding plots below.
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


def _crosscheck_file(
    results_dir: str | Path,
    *,
    task: str,
    model: str,
    response: str,
    train_predictor: str,
    test_predictor: str,
    window: int,
) -> Path:
    """Find one DEF-trained/cross-tested result CSV."""
    directory = Path(results_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {directory}")

    expected = (
        f"crossCheck_{task}_raw_metrics_{model}-{response}-{train_predictor}-"
        f"{window}_on_{response}_{test_predictor}.csv"
    )
    # Some legacy Logreg nowcast files use underscores between model,
    # response, predictor, and window.  Accept that exact filename variant
    # without broadening the match to other tasks/configurations.
    alternate = (
        f"crossCheck_{task}_raw_metrics_{model}_{response}_{train_predictor}_"
        f"{window}_on_{response}_{test_predictor}.csv"
    )
    candidate_names = {expected.lower(), alternate.lower()}
    direct = directory / expected
    if direct.is_file():
        return direct
    alternate_path = directory / alternate
    if alternate_path.is_file():
        return alternate_path
    matches = [p for p in directory.glob("*.csv") if p.name.lower() in candidate_names]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find {expected!r} in {directory}. "
        "Check task, response, model, and train/test predictor labels."
    )


def _read_bootstrap_csv(
    path: Path,
    *,
    window: int,
    period_labels: Mapping[str, str] | None,
) -> pd.DataFrame:
    """Read one metric CSV into tidy bootstrap-replicate data."""
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
    return tidy[["period", "window", "metric", "bootstrap", "value"]]


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
        frames.append(
            _read_bootstrap_csv(
                path, window=int(window), period_labels=period_labels,
            )
        )
    return pd.concat(frames, ignore_index=True)


def read_crosscheck_bootstrap_metrics(
    results_dir: str | Path = "results",
    *,
    task: str,
    response: str,
    model: str = "LSTM",
    train_predictor: str = "DEF",
    test_predictor: str = "NRT",
    windows: Sequence[int] = DEFAULT_WINDOWS,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
) -> pd.DataFrame:
    """Read DEF-trained/cross-tested CSVs into tidy bootstrap data.

    The expected filename is
    ``crossCheck_{task}_raw_metrics_{model}-{response}-DEF-{window}_on_{response}_NRT.csv``.
    The returned columns are ``period``, ``window``, ``metric``, ``bootstrap``,
    and ``value``.
    """
    frames: list[pd.DataFrame] = []
    for window in windows:
        path = _crosscheck_file(
            results_dir, task=task, model=model, response=response,
            train_predictor=train_predictor, test_predictor=test_predictor,
            window=int(window),
        )
        frames.append(
            _read_bootstrap_csv(
                path, window=int(window), period_labels=period_labels,
            )
        )
    return pd.concat(frames, ignore_index=True)


def read_absolute_transfer_metrics(
    results_dir: str | Path = "results",
    *,
    task: str,
    response: str,
    model: str = "LSTM",
    windows: Sequence[int] = DEFAULT_WINDOWS,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
) -> pd.DataFrame:
    """Return absolute baseline and transfer scores in one tidy table.

    ``baseline`` is DEF-trained/DEF-tested, while ``transfer`` is
    DEF-trained/NRT-tested. The two rows for a given period/window/metric/
    bootstrap index use the same saved DEF model replicate.
    """
    baseline = read_bootstrap_metrics(
        results_dir, task=task, response=response, predictor="DEF", model=model,
        windows=windows, period_labels=period_labels,
    ).assign(
        response=response,
        configuration="DEF-trained/DEF-tested",
    )
    transfer = read_crosscheck_bootstrap_metrics(
        results_dir, task=task, response=response, model=model,
        windows=windows, period_labels=period_labels,
    ).assign(
        response=response,
        configuration="DEF-trained/NRT-tested",
    )
    return pd.concat([baseline, transfer], ignore_index=True)


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
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 8,
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


def plot_absolute_transfer_comparison(
    results_dir: str | Path = "results",
    output_path: str | Path | None = None,
    *,
    task: str = "Task1",
    response: str | None = None,
    responses: Sequence[str] | None = None,
    model: str = "LSTM",
    metrics: Sequence[str] = ("TSS", "F1"),
    windows: Sequence[int] = DEFAULT_WINDOWS,
    periods: Sequence[str] = DEFAULT_PERIOD_ORDER,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
    dpi: int = 300,
    title: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot absolute DEF baseline and DEF-to-NRT transfer performance.

    The baseline is read from the standard
    ``{task}_raw_metrics_{model}-{response}-DEF-{window}.csv`` files. The
    transfer scores are read from the corresponding ``crossCheck`` files.
    At each lead-time position, one box is drawn for each selected response
    and evaluation type. The default is four boxes: S baseline, S transfer,
    O baseline, and O transfer.

    Box interiors show the interquartile range; whiskers span the 2.5th to
    97.5th percentiles of the saved bootstrap-model replicates. A connector
    is drawn only between the DEF-tested and NRT-tested boxes for the same
    response and lead time. No line connects different forecast hours.

    ``response="S"`` is retained as a compatibility shorthand for a
    single-response plot. Use ``responses=("S", "O")`` for the combined
    figure. The returned table is the tidy absolute-performance data used by
    the plot.
    """
    metric_order = list(metrics)
    if not metric_order or len(set(metric_order)) != len(metric_order):
        raise ValueError("metrics must be a non-empty sequence of unique names")
    if not windows:
        raise ValueError("windows must be non-empty")
    if not periods:
        raise ValueError("periods must be non-empty")

    if response is not None and responses is not None:
        raise ValueError("Use either response or responses, not both")
    if responses is None:
        response_order = [response] if response is not None else ["S", "O"]
    else:
        response_order = list(responses)
    if not response_order or len(set(response_order)) != len(response_order):
        raise ValueError("responses must be a non-empty sequence of unique names")

    absolute = pd.concat(
        [
            read_absolute_transfer_metrics(
                results_dir, task=task, response=selected_response, model=model,
                windows=windows, period_labels=period_labels,
            )
            for selected_response in response_order
        ],
        ignore_index=True,
    )
    absolute = absolute[absolute["metric"].isin(metric_order)].copy()
    period_order = [period for period in periods if period in set(absolute["period"])]
    if not period_order:
        raise ValueError(f"None of the requested periods are present: {list(periods)}")
    missing_metrics = sorted(set(metric_order).difference(absolute["metric"].unique()))
    if missing_metrics:
        raise ValueError(f"Requested metrics are not present: {missing_metrics}")
    absolute = absolute[absolute["period"].isin(period_order)].copy()

    configuration_order = ["DEF-trained/DEF-tested", "DEF-trained/NRT-tested"]
    # Academic-figure-skill categorical palette.  Response is encoded by
    # colour; evaluation type is encoded redundantly by fill/hatching.
    response_colors = {
        "S": CATEGORICAL[0],  # publication-safe blue
        "O": CATEGORICAL[3],  # publication-safe orange
    }
    response_display_names = {
        "S": "Science Quality",
        "O": "SWPC-FTP",
    }
    missing_colors = [selected_response for selected_response in response_order
                      if selected_response not in response_colors]
    if missing_colors:
        raise ValueError(
            f"No publication-safe color is defined for response(s): {missing_colors}. "
            "Use S and/or O."
        )
    box_order = [
        (selected_response, configuration)
        for selected_response in response_order
        for configuration in configuration_order
    ]
    x = np.arange(len(windows), dtype=float)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(
        len(metric_order), len(period_order),
        figsize=(3.65 * len(period_order), 2.75 * len(metric_order)),
        sharex="row", sharey="row", squeeze=False,
    )
    # Reserve a compact top band for one legend row so it cannot collide with
    # panel titles.  There is intentionally no overall figure title.
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.84,
                        wspace=0.12, hspace=0.28)

    for row_index, metric in enumerate(metric_order):
        metric_data = absolute[absolute["metric"] == metric]
        row_values = metric_data["value"].to_numpy(float)
        if metric.upper() in {"F1", "POD", "ACC"}:
            row_limits = (0.0, 1.0)
        else:
            lower = float(np.nanmin(row_values))
            upper = float(np.nanmax(row_values))
            margin = max(0.03, 0.08 * (upper - lower if upper > lower else 1.0))
            row_limits = (min(0.0, lower - margin), max(1.0, upper + margin))

        for column_index, period in enumerate(period_order):
            ax = axes[row_index, column_index]
            panel = metric_data[metric_data["period"] == period]
            # Four boxes at each lead time when both responses are selected;
            # keep the pair compact while retaining a visible gap between pairs.
            if len(box_order) == 4:
                offsets = np.array([-0.27, -0.09, 0.09, 0.27])
            else:
                offsets = np.linspace(-0.13, 0.13, len(box_order))
            box_width = min(0.12, 0.62 / max(len(box_order), 1))
            medians: dict[tuple[str, str, int], float] = {}
            for box_index, (selected_response, configuration) in enumerate(box_order):
                values_by_window = []
                for window in windows:
                    values = panel.loc[
                        (panel["response"] == selected_response)
                        & (panel["configuration"] == configuration)
                        & (panel["window"] == window),
                        "value",
                    ].to_numpy(float)
                    if not values.size:
                        raise ValueError(
                            f"Missing {metric} result for {task}, {selected_response}, "
                            f"{period}, {configuration}, or window {window}"
                        )
                    values_by_window.append(values)
                    medians[(selected_response, configuration, int(window))] = float(
                        np.median(values)
                    )

                color = response_colors[selected_response]
                boxes = ax.boxplot(
                    values_by_window,
                    positions=x + offsets[box_index],
                    widths=box_width,
                    patch_artist=True,
                    manage_ticks=False,
                    whis=(2.5, 97.5),
                    showfliers=False,
                    medianprops={"color": "#202020", "linewidth": 1.0},
                    whiskerprops={"color": "#4A4A4A", "linewidth": 0.7},
                    capprops={"color": "#4A4A4A", "linewidth": 0.7},
                    boxprops={"edgecolor": "#222222", "linewidth": 0.75},
                )
                for box in boxes["boxes"]:
                    if configuration == configuration_order[0]:
                        box.set_facecolor(color)
                        box.set_edgecolor("#222222")
                        box.set_alpha(0.92)
                    else:
                        # Avoid the previous low-contrast translucent fill:
                        # NRT-tested boxes are white, hatched, and outlined
                        # with the response colour.
                        box.set_facecolor("white")
                        box.set_edgecolor(color)
                        box.set_alpha(1.0)
                        box.set_hatch("///")

            # Connect only the two paired evaluations at each lead time. These
            # short segments do not imply a trend across forecasting hours.
            for selected_response in response_order:
                pair_color = response_colors[selected_response]
                for window_index, window in enumerate(windows):
                    y_pair = [
                        medians[(selected_response, configuration_order[0], int(window))],
                        medians[(selected_response, configuration_order[1], int(window))],
                    ]
                    x_pair = [
                        x[window_index] + offsets[box_order.index((selected_response, configuration_order[0]))],
                        x[window_index] + offsets[box_order.index((selected_response, configuration_order[1]))],
                    ]
                    ax.plot(x_pair, y_pair, color=pair_color, linewidth=0.9, zorder=4)

            ax.set_ylim(*row_limits)
            ax.set_xticks(x, ["Nowcast" if int(window) == 0 else f"{window} h" for window in windows])
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.75, zorder=0)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=7, direction="out", length=3)
            ax.set_title(
                f"{chr(97 + row_index * len(period_order) + column_index)}  "
                f"{metric}, {period}", loc="left", fontweight="bold", pad=5,
            )
            if column_index == 0:
                ax.set_ylabel(f"{metric} score")
            if row_index == len(metric_order) - 1:
                ax.set_xlabel("Forecast lead time")

    # ``title`` is accepted for API compatibility but intentionally ignored:
    # the manuscript figure should not carry an overall title.
    # One compact legend row combines the response name and test pairing for
    # each visual style, avoiding the previous two-row legend arrangement.
    legend_handles = []
    for selected_response, configuration in box_order:
        is_baseline = configuration == configuration_order[0]
        color = response_colors[selected_response]
        legend_handles.append(
            Patch(
                facecolor=color if is_baseline else "white",
                edgecolor="#222222" if is_baseline else color,
                linewidth=0.7,
                hatch=None if is_baseline else "///",
                alpha=0.92 if is_baseline else 1.0,
                label=(f"{response_display_names[selected_response]} "
                       f"({'DEF/DEF' if is_baseline else 'DEF/NRT'})"),
            )
        )
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=len(legend_handles),
        frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.955),
        columnspacing=0.85, handlelength=1.25, handletextpad=0.4,
    )

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, absolute


def paired_absolute_transfer_differences(
    results_dir: str | Path = "results",
    *,
    task: str,
    response: str,
    model: str = "LSTM",
    windows: Sequence[int] = DEFAULT_WINDOWS,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
) -> pd.DataFrame:
    """Return matched cross-domain score changes for one configuration.

    The difference is ``DEF-trained/NRT-tested - DEF-trained/DEF-tested``.
    Baseline and cross-domain rows are paired by period, forecast window,
    metric, and bootstrap-model replicate. A positive value therefore means
    that the saved DEF-trained model scored higher on the NRT test set; it is
    not, by itself, proof of successful transfer because the two test domains
    may differ in difficulty or class composition.
    """
    absolute = read_absolute_transfer_metrics(
        results_dir, task=task, response=response, model=model,
        windows=windows, period_labels=period_labels,
    )
    keys = ["period", "window", "metric", "bootstrap"]
    baseline_name = "DEF-trained/DEF-tested"
    cross_domain_name = "DEF-trained/NRT-tested"
    frames: dict[str, pd.DataFrame] = {}
    for configuration, output_column in (
        (baseline_name, "def_tested"),
        (cross_domain_name, "nrt_tested"),
    ):
        frame = absolute.loc[
            absolute["configuration"] == configuration, keys + ["value"]
        ].rename(columns={"value": output_column})
        duplicate = frame.duplicated(keys, keep=False)
        if duplicate.any():
            examples = frame.loc[duplicate, keys].head(3).to_dict("records")
            raise ValueError(
                f"Duplicate pairing keys for {configuration}: {examples}"
            )
        frames[configuration] = frame

    paired = frames[baseline_name].merge(
        frames[cross_domain_name], on=keys, how="outer", validate="one_to_one",
        indicator=True,
    )
    if not (paired["_merge"] == "both").all():
        unmatched = paired.loc[paired["_merge"] != "both", keys + ["_merge"]].head(8)
        raise ValueError(
            "DEF-tested and NRT-tested results do not share identical paired "
            f"bootstrap keys. Examples: {unmatched.to_dict('records')}"
        )
    paired = paired.drop(columns="_merge")
    paired["difference"] = paired["nrt_tested"] - paired["def_tested"]
    paired["task"] = task
    paired["response"] = response
    paired["model"] = model
    return paired[
        keys + ["task", "response", "model", "def_tested", "nrt_tested", "difference"]
    ]


def plot_absolute_transfer_difference_comparison(
    results_dir: str | Path = "results",
    output_path: str | Path | None = (
        "figure/absolute_transfer/absolute_transfer_difference_comparison.pdf"
    ),
    *,
    tasks: Sequence[str] = ("Task1",),
    responses: Sequence[str] = ("S", "O"),
    models: Sequence[str] = ("LSTM", "Logreg"),
    metrics: Sequence[str] = ("TSS", "FAR"),
    windows: Sequence[int] = DEFAULT_WINDOWS,
    periods: Sequence[str] = DEFAULT_PERIOD_ORDER,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
    dpi: int = 300,
    title: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot paired cross-domain score changes for models within one task.

    Every point is the median of matched bootstrap-model differences defined
    as ``DEF-trained/NRT-tested - DEF-trained/DEF-tested``. Error bars span the
    2.5th to 97.5th percentiles. Colour identifies model, marker shape identifies
    task when more than one is requested, and filled/open markers identify
    Science Quality/SWPC-FTP responses. The default plots Task1 only; use
    :func:`write_absolute_transfer_difference_suite` to write separate figures
    for Task1 and Task2.

    The default layout contains one row per metric and one column per disjoint
    testing period. If ``output_path`` is a PDF or SVG, a 300-dpi PNG preview is
    written beside the vector master. The returned table contains every paired
    bootstrap difference used in the plot.
    """
    selections = {
        "tasks": list(tasks), "responses": list(responses),
        "models": list(models), "metrics": list(metrics),
        "windows": list(windows), "periods": list(periods),
    }
    for name, values in selections.items():
        if not values:
            raise ValueError(f"{name} must be non-empty")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must contain unique values")
    if len(responses) > 2:
        raise ValueError("At most two responses can be distinguished by marker fill")

    differences = pd.concat(
        [
            paired_absolute_transfer_differences(
                results_dir, task=task, response=response, model=model,
                windows=windows, period_labels=period_labels,
            )
            for task in tasks
            for model in models
            for response in responses
        ],
        ignore_index=True,
    )
    differences = differences[
        differences["metric"].isin(metrics)
        & differences["period"].isin(periods)
    ].copy()
    missing_metrics = sorted(set(metrics).difference(differences["metric"].unique()))
    missing_periods = sorted(set(periods).difference(differences["period"].unique()))
    if missing_metrics:
        raise ValueError(f"Requested metrics are not present: {missing_metrics}")
    if missing_periods:
        raise ValueError(f"Requested periods are not present: {missing_periods}")

    metric_order = list(metrics)
    period_order = list(periods)
    window_order = _present_order(differences["window"], windows)
    task_order = list(tasks)
    response_order = list(responses)
    model_order = list(models)
    model_colors = {
        model: CATEGORICAL[index % len(CATEGORICAL)]
        for index, model in enumerate(model_order)
    }
    task_markers = {
        task: ("o", "s", "^", "D")[index % 4]
        for index, task in enumerate(task_order)
    }
    response_names = {"S": "Science Quality", "O": "SWPC-FTP"}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(
        len(metric_order), len(period_order),
        figsize=(7.2, 2.35 * len(metric_order) + 0.65),
        sharex=True, sharey="row", squeeze=False,
    )
    fig.subplots_adjust(
        left=0.10, right=0.985, bottom=0.12, top=0.84,
        wspace=0.12, hspace=0.28,
    )
    x_centres = np.arange(len(window_order), dtype=float)
    combinations = [
        (model, task, response)
        for model in model_order
        for task in task_order
        for response in response_order
    ]
    offsets = (
        np.linspace(-0.30, 0.30, len(combinations))
        if len(combinations) > 1 else np.array([0.0])
    )

    summaries = (
        differences.groupby(
            ["metric", "period", "window", "model", "task", "response"],
            observed=True,
        )["difference"]
        .agg(
            median="median",
            low=lambda values: values.quantile(0.025),
            high=lambda values: values.quantile(0.975),
            n="size",
        )
        .reset_index()
    )
    for row_index, metric in enumerate(metric_order):
        metric_summary = summaries[summaries["metric"] == metric]
        interval_extent = float(
            np.nanmax(np.abs(metric_summary[["low", "high"]].to_numpy(float)))
        )
        y_limit = max(0.05, interval_extent * 1.10)
        for column_index, period in enumerate(period_order):
            ax = axes[row_index, column_index]
            panel = metric_summary[metric_summary["period"] == period]
            ax.axhline(0, color="#555555", linestyle="--", linewidth=0.7, zorder=1)
            ax.grid(
                axis="y", color="#E0E0E0", linewidth=0.3, alpha=0.5, zorder=0,
            )
            ax.set_axisbelow(True)
            for combination_index, (model, task, response) in enumerate(combinations):
                selected = (
                    panel[
                        (panel["model"] == model)
                        & (panel["task"] == task)
                        & (panel["response"] == response)
                    ]
                    .set_index("window")
                    .reindex(window_order)
                )
                median = selected["median"].to_numpy(float)
                low = selected["low"].to_numpy(float)
                high = selected["high"].to_numpy(float)
                color = model_colors[model]
                filled = response_order.index(response) == 0
                ax.errorbar(
                    x_centres + offsets[combination_index], median,
                    yerr=np.vstack((median - low, high - median)),
                    fmt=task_markers[task], linestyle="none", markersize=4.2,
                    color=color, markerfacecolor=color if filled else "white",
                    markeredgecolor=color, markeredgewidth=0.8,
                    elinewidth=0.8, capsize=1.8, capthick=0.7, zorder=3,
                )
            panel_label = chr(97 + row_index * len(period_order) + column_index)
            ax.set_title(
                f"{panel_label}  {metric}, {period}", loc="left",
                fontweight="bold", pad=5,
            )
            ax.set_ylim(-y_limit, y_limit)
            ax.set_xticks(
                x_centres,
                ["Nowcast" if window == 0 else f"{window} h" for window in window_order],
            )
            if column_index == 0:
                ax.set_ylabel("Cross-domain score change\n(NRT-tested - DEF-tested)")
            if row_index == len(metric_order) - 1:
                ax.set_xlabel("Forecast lead time")
            ax.margins(x=0.08)

    model_handles = [
        Line2D([], [], color=model_colors[model], marker="o", linestyle="none",
               markersize=4.5, label=model)
        for model in model_order
    ]
    task_handles = [
        Line2D([], [], color="#333333", marker=task_markers[task],
               markerfacecolor="#333333", linestyle="none", markersize=4.5,
               label=task)
        for task in task_order
    ]
    response_handles = [
        Line2D([], [], color="#333333", marker="o", linestyle="none", markersize=4.5,
               markerfacecolor="#333333" if index == 0 else "white",
               label=response_names.get(response, response))
        for index, response in enumerate(response_order)
    ]
    fig.legend(
        handles=model_handles, loc="upper left", bbox_to_anchor=(0.09, 0.925),
        ncol=len(model_handles), columnspacing=1.0, handletextpad=0.35,
    )
    if len(task_handles) > 1:
        fig.legend(
            handles=task_handles, loc="upper center", bbox_to_anchor=(0.50, 0.965),
            ncol=len(task_handles), columnspacing=1.0, handletextpad=0.35,
        )
    fig.legend(
        handles=response_handles, loc="upper right", bbox_to_anchor=(0.99, 0.925),
        ncol=len(response_handles), columnspacing=1.0, handletextpad=0.35,
    )
    figure_title = title if title is not None else (
        task_order[0] if len(task_order) == 1 else None
    )
    if figure_title:
        fig.suptitle(figure_title, y=0.975, fontweight="bold")

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=dpi, facecolor="white")
        if output.suffix.lower() in {".pdf", ".svg", ".eps"}:
            fig.savefig(
                output.with_suffix(".png"), bbox_inches="tight", dpi=dpi,
                facecolor="white",
            )
    return fig, differences


def write_absolute_transfer_difference_suite(
    results_dir: str | Path = "results",
    output_directory: str | Path = "figure/absolute_transfer",
    *,
    tasks: Sequence[str] = ("Task1", "Task2"),
    responses: Sequence[str] = ("S", "O"),
    models: Sequence[str] = ("LSTM", "Logreg"),
    metrics: Sequence[str] = ("TSS", "FAR"),
    windows: Sequence[int] = DEFAULT_WINDOWS,
    periods: Sequence[str] = DEFAULT_PERIOD_ORDER,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
    dpi: int = 300,
) -> dict[str, tuple[plt.Figure, pd.DataFrame]]:
    """Write one paired cross-domain difference figure per task.

    Each task receives its own vector PDF and 300-dpi PNG preview. The returned
    dictionary maps the task label to the corresponding figure and complete
    paired-bootstrap table.
    """
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("tasks must be a non-empty sequence of unique names")
    output_directory = Path(output_directory)
    outputs: dict[str, tuple[plt.Figure, pd.DataFrame]] = {}
    for task in tasks:
        output_path = output_directory / f"{task}_absolute_transfer_difference.pdf"
        outputs[task] = plot_absolute_transfer_difference_comparison(
            results_dir=results_dir,
            output_path=output_path,
            tasks=(task,),
            responses=responses,
            models=models,
            metrics=metrics,
            windows=windows,
            periods=periods,
            period_labels=period_labels,
            dpi=dpi,
        )
    return outputs


def write_absolute_transfer_suite(
    results_dir: str | Path = "results",
    output_directory: str | Path = "figure/absolute_transfer",
    *,
    tasks: Sequence[str] = ("Task1", "Task2"),
    responses: Sequence[str] = ("S", "O"),
    model: str = "LSTM",
    metrics: Sequence[str] = ("TSS", "F1"),
    windows: Sequence[int] = DEFAULT_WINDOWS,
    periods: Sequence[str] = DEFAULT_PERIOD_ORDER,
    period_labels: Mapping[str, str] | None = DEFAULT_PERIOD_LABELS,
    dpi: int = 300,
) -> dict[tuple[str, str], tuple[plt.Figure, pd.DataFrame]]:
    """Write one combined absolute-performance plot for every task."""
    output_directory = Path(output_directory)
    outputs: dict[tuple[str, str], tuple[plt.Figure, pd.DataFrame]] = {}
    for task in tasks:
        response_label = "_".join(responses)
        output_path = output_directory / f"{task}_{model}_{response_label}_absolute_transfer.pdf"
        outputs[(task, response_label)] = plot_absolute_transfer_comparison(
            results_dir=results_dir, output_path=output_path,
            task=task, responses=responses, model=model, metrics=metrics,
            windows=windows, periods=periods, period_labels=period_labels,
            dpi=dpi,
        )
    return outputs
