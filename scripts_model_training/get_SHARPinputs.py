from pathlib import Path
import numpy as np
import pandas as pd

params = [
    'USFLUX','MEANGAM','MEANGBT','MEANGBZ','MEANGBH','MEANJZD',
    'TOTUSJZ','MEANALP','MEANJZH','TOTUSJH','ABSNJZH','SAVNCPP',
    'MEANPOT','TOTPOT','MEANSHR','SHRGT45','SIZE','SIZE_ACR',
    'NACR','NPIX'
]

def read_all_harp_csvs(root: str | Path, recursive: bool = False) -> dict[int, pd.DataFrame]:
    """
    Read all per-HARP CSVs under `root` and return {harpnum: DataFrame} with
    T_REC parsed to datetime and sorted ascending.
    Assumes files are named like HARP_XXXX.csv
    """
    root = Path(root)
    pattern = "**/HARP_*.csv" if recursive else "HARP_*.csv"
    harps: dict[int, pd.DataFrame] = {}

    for f in root.glob(pattern):
        try:
            harpnum = int(f.stem.split("_")[1])
        except Exception:
            continue

        df = pd.read_csv(f)
        print(f"Reading {f.name} with {len(df)} rows")

        if len(df) <= 120:
            print(f"  Skipping HARP {harpnum}")
            continue

        # Normalize T_REC
        dt = df["T_REC"].astype(str).str.replace("_TAI", "", regex=False)
        df["T_REC"] = pd.to_datetime(dt, format=None, errors="coerce")
        #df = df.dropna(subset=["T_REC"]).sort_values("T_REC").reset_index(drop=True)

        harps[harpnum] = df
    print(f"Loaded {len(harps)} HARPs")
    return harps

def generate_inputs_for_one_harp(harp: pd.DataFrame,
                                 params: list[str],
                                 max_samples: int = 5000,
                                 window_hours: int = 24,
                                 stride_hours: int = 1):
    """
    STRICT version: expects perfect 12-min cadence (5 points/hour).
    Builds 24h windows (size=120) with 1h stride, stacks to (N, 120, P).
    Skips any window that isn't exactly 120 rows (shouldn't happen if cadence is enforced).
    Returns: (np.ndarray[N, 120, P], list[pd.Timestamp] observation end-times)
    """
    harp = harp.sort_values("T_REC").reset_index(drop=True)
    if "T_REC" not in harp.columns:
        raise ValueError("HARP dataframe must contain 'T_REC'.")

    missing = [p for p in params if p not in harp.columns]
    if missing:
        raise ValueError(f"Missing parameter columns in HARP: {missing}")

    # Constants for 12-min cadence
    pts_per_hour = 5
    window_len = window_hours * pts_per_hour  # 24h * 5 = 120
    stride = stride_hours * pts_per_hour      # 1h * 5 = 5

    # Create a time-indexed view to slice by time if desired (we’ll use index math)
    times = harp["T_REC"].to_numpy()
    Xall = harp[params].to_numpy()

    inputs = []
    end_times = []

    # Find the index where a 24h window ends
    # We slide start by `stride` rows (i.e., 1h) each step
    start_idx = 0
    last_idx = len(harp) - 1

    while start_idx + window_len - 1 <= last_idx and len(inputs) < max_samples:
        end_idx = start_idx + window_len  # exclusive
        window = Xall[start_idx:end_idx]
        if window.shape[0] == window_len:
            inputs.append(window)
            end_times.append(times[end_idx - 1])  # obs time = window end
        start_idx += stride

    print(f"Generated {len(inputs)} samples (120 rows per window).")
    if len(inputs) == 0:
        return np.empty((0, window_len, len(params))), []

    return np.stack(inputs, axis=0), end_times

def generate_inputs_for_all_harps(harps: dict[int, pd.DataFrame],
                                  params: list[str],
                                  max_samples_per_harp: int = 5000,
                                  window_hours: int = 24,
                                  stride_hours: int = 1):
    """
    Apply strict generator to every HARP.
    Returns {harpnum: (inputs[N,120,P], times[List[pd.Timestamp]])}
    """
    out = {}
    for harpnum, df in harps.items():
        ar = df['NOAA_AR'].unique()
        try:
            X, t = generate_inputs_for_one_harp(
                df, params=params,
                max_samples=max_samples_per_harp,
                window_hours=window_hours,
                stride_hours=stride_hours
            )
            if X.shape[0] > 0:
                out[harpnum] = (X, t, ar)
        except Exception as e:
            print(f"HARP {harpnum} skipped: {e}")
    return out