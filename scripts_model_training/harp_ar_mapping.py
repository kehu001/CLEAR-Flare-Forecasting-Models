import re
import io
import os
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

JSOC_HARP_NOAA_URL = "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"

def _extract_ints(cell: str) -> list[int]:
    """Pull all integer NOAA AR IDs from a messy cell (e.g., 'ARs: 12738,12739')."""
    return list(map(int, re.findall(r"\d+", str(cell))))

def fetch_harp_noaa_map(url: str = JSOC_HARP_NOAA_URL,
                        timeout: float = 30.0) -> pd.DataFrame:
    """
    Download and parse the JSOC HARP→NOAA AR mapping table.

    Returns a tidy DataFrame with:
      - HARPNUM (int)
      - NOAA_AR (int)    # exploded: one row per (HARPNUM, NOAA_AR)
      - any passthrough columns if readable (e.g., time ranges), kept as-is
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    # Try pandas parsing first (many of JSOC text tables are whitespace separated)
    raw = io.StringIO(resp.text)
    try:
        df = pd.read_csv(
            raw,
            sep=r"\s+",
            engine="python",
            comment="#",
            dtype=str
        )
    except Exception:
        # fallback: clean lines and split
        lines = [ln for ln in resp.text.splitlines() if ln.strip() and not ln.startswith("#")]
        parts = [re.split(r"\s+", ln.strip()) for ln in lines]
        # Heuristic: first row is header if it has non-numeric tokens
        header_like = any(not tok.isdigit() for tok in parts[0])
        if header_like:
            cols = parts[0]
            body = parts[1:]
        else:
            # Generic columns if no header present
            cols = [f"col{i}" for i in range(len(parts[0]))]
            body = parts
        df = pd.DataFrame(body, columns=cols)

    # Normalize likely column names
    cols_lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns={c: c.upper() for c in df.columns})

    # Identify the HARP and NOAA columns by common names/heuristics
    harp_col = next((c for c in df.columns if c.upper() in {"HARP", "HARPNUM", "HARPS", "HARPNUMBER"}), None)
    noaa_col = next((c for c in df.columns if "NOAA" in c.upper()), None)

    if harp_col is None:
        # Try first numeric-looking column as HARPNUM
        for c in df.columns:
            if df[c].astype(str).str.fullmatch(r"\d+").all():
                harp_col = c
                break
    if harp_col is None:
        raise ValueError("Could not find HARPNUM column in the JSOC table.")

    if noaa_col is None:
        # If a column literally contains the words AR/NOAA, prefer it
        for c in df.columns:
            if re.search(r"NOAA|AR", c, re.IGNORECASE):
                noaa_col = c
                break
    if noaa_col is None:
        # As a last resort, assume the last column contains ARs
        noaa_col = df.columns[-1]

    # Clean and explode NOAA list into one row per (HARPNUM, NOAA_AR)
    df["_HARPNUM"] = pd.to_numeric(df[harp_col], errors="coerce").astype("Int64")
    df["_NOAA_LIST"] = df[noaa_col].map(_extract_ints)
    # Drop rows with no HARP id
    df = df.dropna(subset=["_HARPNUM"])

    exploded = df.explode("_NOAA_LIST", ignore_index=True)
    exploded = exploded.rename(columns={"_HARPNUM": "HARPNUM", "_NOAA_LIST": "NOAA_AR"})
    exploded["NOAA_AR"] = exploded["NOAA_AR"].astype("Int64")

    # Keep original columns (optional) but move HARPNUM/NOAA_AR front
    front = ["HARPNUM", "NOAA_AR"]
    other = [c for c in exploded.columns if c not in front]
    exploded = exploded[front + other]

    # Deduplicate just in case
    exploded = exploded.drop_duplicates(subset=["HARPNUM", "NOAA_AR"]).reset_index(drop=True)
    # delete the duplicate columns if any
    exploded = exploded.loc[:,~exploded.columns.duplicated()]

    return exploded


def save_harp_noaa_map(csv_path: str | os.PathLike,
                       url: str = JSOC_HARP_NOAA_URL,
                       overwrite: bool = True,
                       timeout: float = 30.0) -> pd.DataFrame:
    """
    Fetch the mapping and write to CSV.
    If overwrite=True, always replace the file (your requirement).
    Returns the DataFrame written.
    """
    df = fetch_harp_noaa_map(url=url, timeout=timeout)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite on each run
    df.to_csv(csv_path, index=False)
    return df
