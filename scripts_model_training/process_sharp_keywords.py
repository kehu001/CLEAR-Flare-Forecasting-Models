#!/usr/bin/env python3
"""
process_sharp_keywords.py

1) lets the user choose start and end times,
2) keeps only rows within ±70 deg longitude,
3) splits data by HARP number and saves each HARP to its own CSV,
4) within each HARP, ensures a 12-minute cadence; linearly interpolates missing rows
   for 20 specified SHARP parameters ONLY,
5) on subsequent runs, appends/merges into the same per-HARP CSV (de-duplicated by T_REC).

USAGE EXAMPLES
--------------
# TAI format (recommended to match downloadHMI.py)
python process_sharp_keywords.py --start 2025.11.07_01:00:00_TAI --end 2025.11.07_04:00:00_TAI

# ISO format is also accepted; it will be converted to TAI string internally
python process_sharp_keywords.py --start 2025-11-07T01:00:00 --end 2025-11-07T04:00:00

REQUIREMENTS
------------
- pandas, numpy
- The local `downloadHMI.py` with:
    - SHARP_KEYS list
    - downloadHMIFiles(seriesName, harpTimeStart, harpTimeEnd) function (drms-based)
    - a global `config` module/obj holding `outputPath` (used by downloadHMI.py)
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Import the user's downloader (must be in PYTHONPATH or same directory)
import downloadHMI # expects: SHARP_KEYS, downloadHMIFiles(), config

# ----- Configuration -----
# SHARP series to query (NRT examples)
SHARP_SERIES = 'hmi.sharp_cea_720s_nrt' #, 'hmi.sharp_720s_nrt')

# The 20 SHARP parameters used as features
INTERP_COLS_20 = [
    'USFLUX','MEANGAM','MEANGBT','MEANGBZ','MEANGBH','MEANJZD',
    'TOTUSJZ','MEANALP','MEANJZH','TOTUSJH','ABSNJZH','SAVNCPP',
    'MEANPOT','TOTPOT','MEANSHR','SHRGT45','SIZE','SIZE_ACR',
    'NACR','NPIX'
]


LON_MIN_COL = 'LON_MIN'
LON_MAX_COL = 'LON_MAX'

TIME_COL = 'T_REC'  


def ensure_tai_string(s: str) -> str:
    """
    Accept either 'YYYY.MM.DD_HH:MM:SS_TAI' or ISO ('YYYY-MM-DDTHH:MM:SS').
    Return TAI-style string for DRMS queries and consistent timestamp parsing.
    """
    if s.endswith('_TAI') and len(s) >= 22:
        return s  # seems already in TAI form
    # Try ISO or other common forms
    try:
        dt = datetime.fromisoformat(s.replace('Z',''))
    except ValueError:
        # heuristic for 'YYYY.MM.DD_HH:MM:SS' (no _TAI)
        try:
            dt = datetime.strptime(s, "%Y.%m.%d_%H:%M:%S")
        except ValueError as e:
            raise ValueError(f"Unrecognized time format: {s}") from e
    return dt.strftime("%Y.%m.%d_%H:%M:%S_TAI")


def read_new_csvs(file_path):
    frames = []
    try:
        df = pd.read_csv(file_path)
        frames.append(df)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def filter_longitude(df: pd.DataFrame, lon_abs_max: float = 70.0) -> pd.DataFrame:
    """
    Keep only rows within ±lon_abs_max degrees longitude.
    Here we require both LON_MIN >= -lon_abs_max AND LON_MAX <= lon_abs_max.
    """
    if LON_MIN_COL not in df.columns or LON_MAX_COL not in df.columns:
        # If missing lon info, just return as-is (or drop them)
        return df

    # Coerce to numeric in case of strings
    df[LON_MIN_COL] = pd.to_numeric(df[LON_MIN_COL], errors='coerce')
    df[LON_MAX_COL] = pd.to_numeric(df[LON_MAX_COL], errors='coerce')

    mask = (df[LON_MIN_COL] >= -lon_abs_max) & (df[LON_MAX_COL] <= lon_abs_max)
    return df.loc[mask].copy()


def to_datetime_index(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    """
    Convert T_REC to pandas datetime and set as index.
    """
    if time_col not in df.columns:
        raise ValueError(f"Expected time column '{time_col}' not found in DataFrame.")

    # Convert TAI-like strings "YYYY.MM.DD_HH:MM:SS_TAI" to datetime (treat as naive)
    # Strip trailing "_TAI" if present
    dt_series = df[time_col].astype(str).str.replace('_TAI', '', regex=False)
    # Convert from 'YYYY.MM.DD_HH:MM:SS' -> datetime
    dt_series = pd.to_datetime(dt_series, format="%Y.%m.%d_%H:%M:%S", errors='coerce')

    df = df.copy()
    df[time_col] = dt_series
    df = df.dropna(subset=[time_col])
    df = df.set_index(time_col).sort_index()
    return df


def regrid_and_interpolate_12min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to a strict 12-minute grid (from first to last timestamp) and linearly interpolate
    ONLY the 20 key columns. Non-key columns are left as-is
    """
    if df.empty:
        return df

    # Ensure numeric for interpolation columns
    for c in INTERP_COLS_20:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='12min')
    df = df.reindex(full_index)

    # Interpolate the 20 columns
    cols_present = [c for c in INTERP_COLS_20 if c in df.columns]
    if cols_present:
        df[cols_present] = df[cols_present].interpolate(method='time', limit_direction='both')

    return df


def save_by_harp(df_all: pd.DataFrame, base_output_dir: Path) -> None:
    """
    Split by HARPNUM and save each to a dedicated CSV under HMI/SHARP_by_HARP/HARP_<num>.csv
    Merges with existing file if present, de-duplicates by index timestamp, re-grids to 12 min,
    and interpolates only INTERP_COLS_20.
    """
    if 'HARPNUM' not in df_all.columns:
        print("No HARPNUM column found; nothing to save.")
        return

    out_dir = base_output_dir / 'HMI' / 'SHARP_by_HARP'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by HARP
    for harp, g in df_all.groupby('HARPNUM'):
        try:
            # Work on a copy
            g = g.copy()
            # Build per-HARP path
            out_csv = out_dir / f"HARP_{int(harp)}.csv"

            # Convert time to index
            g = to_datetime_index(g, TIME_COL)

            # If an existing file is present, merge with it before regridding
            if out_csv.exists():
                old = pd.read_csv(out_csv)
                # Convert old to datetime index as well
                if TIME_COL not in old.columns:
                    # If previous version used unnamed index, try recovering
                    if 'Unnamed: 0' in old.columns:
                        old.rename(columns={'Unnamed: 0': TIME_COL}, inplace=True)
                    else:
                        # Fallback: create a dummy time col (will be dropped)
                        old[TIME_COL] = pd.NaT
                old = to_datetime_index(old, TIME_COL)
                # Union, drop duplicates by index (keep last)
                merged = pd.concat([old, g], axis=0)
                merged = merged[~merged.index.duplicated(keep='last')]
            else:
                merged = g

            # Regrid to 12 min and interpolate 20 key params
            merged = regrid_and_interpolate_12min(merged)

            # Persist
            merged.reset_index().rename(columns={'index': TIME_COL}).to_csv(out_csv, index=False)
            #print(f"Saved HARP {int(harp)} to {out_csv}")
        except Exception as e:
            print(f"Failed to save HARP {harp}: {e}")


def sharp_download_main(start: str, end: str, lon: float) -> None:
    #parser = argparse.ArgumentParser(description="Fetch and process SHARP keywords by HARP.")
    #parser.add_argument('--start', required=True, help="Start time (TAI 'YYYY.MM.DD_HH:MM:SS_TAI' or ISO 'YYYY-MM-DDTHH:MM:SS').")
    #parser.add_argument('--end', required=True, help="End time (TAI or ISO).")
    #parser.add_argument('--lon', type=float, default=70.0, help="Absolute longitude threshold (default: 70 degrees).")
    #args = parser.parse_args()

    # Normalize to TAI strings for the downloader
    start_tai = ensure_tai_string(start)
    end_tai = ensure_tai_string(end)

    # Call the user's downloader for each SHARP series
    print('-' * 60)
    print(f"Downloading SHARP keywords: {SHARP_SERIES}")
    print('-' * 60)
    try:
        csvpath = downloadHMI.downloadHMIFiles(SHARP_SERIES, start_tai, end_tai)
        # Where the downloader writes CSVs
        base_output_dir = Path(downloadHMI.config.outputPath)

        # Attempt to derive a file stamp prefix from the start time (used in downloader's filename)
        stamp_prefix = start_tai.replace(':', '').replace('_', '').replace('.', '')

        # Read those CSVs
        df = read_new_csvs(csvpath)
        if df.empty:
            print("No CSVs found from downloader; exiting.")
            sys.exit(0)

        # Filter to ±longitude
        df = filter_longitude(df, lon_abs_max=lon)

        # Save by HARP (merging with existing per-HARP CSVs)
        save_by_harp(df, base_output_dir)
    except Exception as e:
        print(f"Downloader failed for {SHARP_SERIES}: {e}")

    


#if __name__ == '__main__':
    #main()
