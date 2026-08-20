"""Utilities for evaluating saved bootstrap models."""

import importlib
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


from New_SampleConstruction import get_samples
from New_lstm2 import DEVICE, lstm
from utilities import ACC, FAR, F1, HSS, POD, TSS, combine, normalize2


_FINAL_NUMBER = re.compile(
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)

#DEFAULT_TESTING_PERIODS = (
 #   ("2020-01-01", "2022-01-01"),
  #  ("2022-01-01", "2023-01-01"),
   # ("2023-01-01", "2025-01-01"),
    #("2020-01-01", "2025-01-01"),
#)

DEFAULT_TESTING_PERIODS = (
    ("2020-01-01", "2022-01-01"),
    ("2022-01-01", "2025-01-01"),
    ("2020-01-01", "2025-01-01"),
)


def _register_torch_serialization_modules() -> None:
    """Attach private tensor-rebuild helpers needed by torch.load."""
    torch_utils = importlib.import_module("torch._utils")
    setattr(torch, "_utils", torch_utils)


def _find_model_and_threshold(
    model_dir: str | Path,
    model_name: str,
    bootstrap_index: int,
) -> tuple[Path, float]:
    """Find one bootstrap checkpoint and parse its filename threshold."""
    model_dir = Path(model_dir)
    matches = sorted(model_dir.glob(f"{model_name}{bootstrap_index}_*.pth"))

    if not matches:
        expected = model_dir / f"{model_name}{bootstrap_index}_<threshold>.pth"
        raise FileNotFoundError(f"No model found matching {expected}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple models found for bootstrap {bootstrap_index}: "
            + ", ".join(str(path) for path in matches)
        )

    path = matches[0]
    match = _FINAL_NUMBER.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse a threshold from the end of {path.name}")

    threshold = float(match.group(1))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Threshold parsed from {path.name} is {threshold}; expected [0, 1]"
        )
    return path, threshold


def _save_metric_csv(
    period_metric_values: list[tuple[str, str, dict[str, np.ndarray]]],
    *,
    model_name: str,
    results_dir: str | Path,
    results_filename: str | None,
) -> Path:
    """Save every testing period in one Metric/Period/Sample_N CSV."""
    rows = []
    for time1, time2, metric_values in period_metric_values:
        for metric, values in metric_values.items():
            row = {"Metric": metric, "Period": f"{time1} to {time2}"}
            row.update(
                {
                    f"Sample_{index + 1}": float(value)
                    for index, value in enumerate(values)
                }
            )
            rows.append(row)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if results_filename is None:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_.")
        if len(period_metric_values) == 1:
            time1, time2, _ = period_metric_values[0]
            period_suffix = f"{time1}_to_{time2}"
        else:
            period_suffix = "all_periods"
        results_filename = f"raw_metrics_{safe_name}_{period_suffix}.csv"
    elif not results_filename.lower().endswith(".csv"):
        results_filename += ".csv"

    output_path = results_dir / results_filename
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def GenMetrics_lstm(
    model_name,
    sample_obj,
    time1=None,
    time2=None,
    model_dir=None,
    purpose=None,
    batch_size=2048,
    n_models=30,
    num_workers=0,
    results_dir="./results",
    results_filename=None,
    testing_periods=None,
):
    """Evaluate saved LSTM bootstraps and save all periods in one CSV.

    A checkpoint is expected to be named
    ``{model_name}{bootstrap_index}_{threshold}.pth``.  Each checkpoint's own
    threshold is parsed from the final number in its filename.

    When ``time1`` and ``time2`` are omitted, the four standard testing periods
    in ``DEFAULT_TESTING_PERIODS`` are evaluated.  Supplying one explicit
    ``time1``/``time2`` pair retains compatibility with older notebook calls.

    Parameters
    ----------
    testing_periods : sequence of (start, end), optional
        Custom periods to evaluate. Do not combine this argument with
        ``time1``/``time2``.
    results_dir : path-like
        Output directory. It is created automatically.
    results_filename : str or None
        Optional CSV filename. When omitted, a name is derived from
        ``model_name`` and the testing period.
    """
    if model_dir is None or purpose is None:
        raise ValueError("model_dir and purpose are required")

    if testing_periods is not None:
        if time1 is not None or time2 is not None:
            raise ValueError(
                "Use either testing_periods or time1/time2, not both"
            )
        periods = tuple((str(start), str(end)) for start, end in testing_periods)
    elif time1 is None and time2 is None:
        periods = DEFAULT_TESTING_PERIODS
    elif time1 is None or time2 is None:
        raise ValueError("time1 and time2 must be supplied together")
    else:
        periods = ((str(time1), str(time2)),)

    if not periods:
        raise ValueError("At least one testing period is required")

    model_specs = [
        _find_model_and_threshold(model_dir, model_name, i)
        for i in range(n_models)
    ]
    thresholds = np.asarray(
        [threshold for _, threshold in model_specs], dtype=np.float64
    )
    print(
        f"Threshold: {thresholds.mean():.3f}"
        f"[{thresholds.min():.3f},{thresholds.max():.3f}]"
    )

    use_cuda = (
        isinstance(DEVICE, torch.device) and DEVICE.type == "cuda"
    ) or str(DEVICE).startswith("cuda")
    period_metric_values = []
    period_results = {}

    for period_index, (period_start, period_end) in enumerate(periods, start=1):
        print(
            f"\nTesting period {period_index}/{len(periods)}: "
            f"{period_start} to {period_end}"
        )
        te_pos_inputs, te_neg_inputs = get_samples(
            sample_obj.inputs_profile,
            sample_obj.labels,
            purpose,
            period_start,
            period_end,
        )

        n_positive = te_pos_inputs.shape[0]
        n_negative = te_neg_inputs.shape[0]
        n_total = n_positive + n_negative
        if n_total == 0:
            raise ValueError(
                f"No test samples found between {period_start} and {period_end}"
            )
        print(f"Positive: {n_positive}, Negative: {n_negative}")
        pos_rate = n_positive / n_total

        te_inputs, te_targets = combine(te_pos_inputs, te_neg_inputs)
        te_inputs = normalize2(te_inputs, te_inputs)
        te_inputs = np.asarray(te_inputs, dtype=np.float32)
        te_targets = np.asarray(te_targets, dtype=np.int64).reshape(-1)

        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(te_inputs)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            drop_last=False,
        )
        input_dim = te_inputs.shape[2]

        pods = np.empty(n_models, dtype=np.float64)
        fars = np.empty(n_models, dtype=np.float64)
        tsss = np.empty(n_models, dtype=np.float64)
        hsss = np.empty(n_models, dtype=np.float64)
        f1s = np.empty(n_models, dtype=np.float64)
        accs = np.empty(n_models, dtype=np.float64)

        for i, (path, threshold) in enumerate(model_specs):
            print(
                f"Bootstrap {i + 1}/{n_models}: "
                f"{path.name}, threshold={threshold:g}"
            )
            _register_torch_serialization_modules()
            state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
            model = lstm(input_dim).to(DEVICE)
            model.load_state_dict(state_dict)
            model.eval()

            if 0.0 < threshold < 1.0:
                logit_threshold = np.log(threshold / (1.0 - threshold))
            elif threshold <= 0.0:
                logit_threshold = -np.inf
            else:
                logit_threshold = np.inf

            predictions = []
            with torch.inference_mode():
                for (batch_inputs,) in test_loader:
                    batch_inputs = batch_inputs.to(
                        DEVICE,
                        dtype=torch.float32,
                        non_blocking=use_cuda,
                    )
                    logits = model(batch_inputs).reshape(-1)
                    predictions.append(
                        (logits > logit_threshold).to(torch.uint8).cpu()
                    )

            y_pred_i = torch.cat(predictions).numpy()
            tsss[i] = TSS(y_pred_i, te_targets)
            pods[i] = POD(y_pred_i, te_targets)
            fars[i] = FAR(y_pred_i, te_targets)
            hsss[i] = HSS(y_pred_i, te_targets)
            f1s[i] = F1(y_pred_i, te_targets)
            accs[i] = ACC(y_pred_i, te_targets)

        metric_values = {
            "TSS": tsss,
            "HSS": hsss,
            "POD": pods,
            "F1": f1s,
            "FAR": fars,
            "ACC": accs,
        }
        for name, values in metric_values.items():
            print(
                f"{name}: {values.mean():.2f}"
                f"[{values.min():.2f},{values.max():.2f}]"
            )

        period_metric_values.append(
            (period_start, period_end, metric_values)
        )
        period_results[f"{period_start} to {period_end}"] = (
            tsss,
            hsss,
            pods,
            f1s,
            fars,
            accs,
            pos_rate,
        )

    output_path = _save_metric_csv(
        period_metric_values,
        model_name=model_name,
        results_dir=results_dir,
        results_filename=results_filename,
    )
    print(f"Saved metrics to {output_path}")

    if len(period_results) == 1:
        # Preserve the original return signature for explicit single-period calls.
        return next(iter(period_results.values()))
    return period_results


def GenMetrics_logreg(
    model_name,
    sample_obj,
    time1=None,
    time2=None,
    model_dir=None,
    purpose=None,
    n_models=30,
    results_dir="./results",
    results_filename=None,
    testing_periods=None,
):
    """Evaluate logistic-regression bootstraps and save all periods in one CSV.

    Checkpoints must be joblib files named
    ``{model_name}{bootstrap_index}_{threshold}.pth``. Each fitted pipeline is
    expected to expose ``predict_proba``; the threshold is parsed from the
    checkpoint filename.

    Omitting ``time1`` and ``time2`` evaluates ``DEFAULT_TESTING_PERIODS``.
    Supplying one explicit pair preserves the original single-period return
    signature.
    """
    if model_dir is None or purpose is None:
        raise ValueError("model_dir and purpose are required")

    if testing_periods is not None:
        if time1 is not None or time2 is not None:
            raise ValueError(
                "Use either testing_periods or time1/time2, not both"
            )
        periods = tuple((str(start), str(end)) for start, end in testing_periods)
    elif time1 is None and time2 is None:
        periods = DEFAULT_TESTING_PERIODS
    elif time1 is None or time2 is None:
        raise ValueError("time1 and time2 must be supplied together")
    else:
        periods = ((str(time1), str(time2)),)

    if not periods:
        raise ValueError("At least one testing period is required")

    model_specs = [
        _find_model_and_threshold(model_dir, model_name, i)
        for i in range(n_models)
    ]
    thresholds = np.asarray(
        [threshold for _, threshold in model_specs], dtype=np.float64
    )
    print(
        f"Threshold: {thresholds.mean():.3f}"
        f"[{thresholds.min():.3f},{thresholds.max():.3f}]"
    )

    period_metric_values = []
    period_results = {}

    for period_index, (period_start, period_end) in enumerate(periods, start=1):
        print(
            f"\nTesting period {period_index}/{len(periods)}: "
            f"{period_start} to {period_end}"
        )
        te_pos_inputs, te_neg_inputs = get_samples(
            sample_obj.inputs_profile,
            sample_obj.labels,
            purpose,
            period_start,
            period_end,
        )

        n_positive = te_pos_inputs.shape[0]
        n_negative = te_neg_inputs.shape[0]
        n_total = n_positive + n_negative
        if n_total == 0:
            raise ValueError(
                f"No test samples found between {period_start} and {period_end}"
            )
        print(f"Positive: {n_positive}, Negative: {n_negative}")
        pos_rate = n_positive / n_total

        te_inputs, te_targets = combine(te_pos_inputs, te_neg_inputs)
        te_inputs = np.asarray(te_inputs, dtype=np.float32)
        x_test = te_inputs.reshape(te_inputs.shape[0], -1)
        te_targets = np.asarray(te_targets, dtype=np.int64).reshape(-1)

        pods = np.empty(n_models, dtype=np.float64)
        fars = np.empty(n_models, dtype=np.float64)
        tsss = np.empty(n_models, dtype=np.float64)
        hsss = np.empty(n_models, dtype=np.float64)
        f1s = np.empty(n_models, dtype=np.float64)
        accs = np.empty(n_models, dtype=np.float64)

        for i, (path, threshold) in enumerate(model_specs):
            print(
                f"Bootstrap {i + 1}/{n_models}: "
                f"{path.name}, threshold={threshold:g}"
            )
            model = joblib.load(path)
            y_probability = model.predict_proba(x_test)[:, 1]
            y_pred_i = (y_probability > threshold).astype(np.uint8)

            tsss[i] = TSS(y_pred_i, te_targets)
            pods[i] = POD(y_pred_i, te_targets)
            fars[i] = FAR(y_pred_i, te_targets)
            hsss[i] = HSS(y_pred_i, te_targets)
            f1s[i] = F1(y_pred_i, te_targets)
            accs[i] = ACC(y_pred_i, te_targets)

        metric_values = {
            "TSS": tsss,
            "HSS": hsss,
            "POD": pods,
            "F1": f1s,
            "FAR": fars,
            "ACC": accs,
        }
        for name, values in metric_values.items():
            print(
                f"{name}: {values.mean():.2f}"
                f"[{values.min():.2f},{values.max():.2f}]"
            )

        period_metric_values.append(
            (period_start, period_end, metric_values)
        )
        period_results[f"{period_start} to {period_end}"] = (
            tsss,
            hsss,
            pods,
            f1s,
            fars,
            accs,
            pos_rate,
        )

    output_path = _save_metric_csv(
        period_metric_values,
        model_name=model_name,
        results_dir=results_dir,
        results_filename=results_filename,
    )
    print(f"Saved metrics to {output_path}")

    if len(period_results) == 1:
        return next(iter(period_results.values()))
    return period_results
