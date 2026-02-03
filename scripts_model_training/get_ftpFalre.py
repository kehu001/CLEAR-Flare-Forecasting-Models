import os
import re
import ftplib
from datetime import datetime
from pathlib import Path
import pandas as pd

"""
Download SWPC operational events summary from FTP open warehouse fron a cut-off date.
Extract XRS flare entries from the downloaded text files into a CSV.
Update csv file on each run (overwrite).
"""

def download_ftp_flares(
    ftp_server="ftp.swpc.noaa.gov",
    ftp_path="/pub/warehouse/2025/2025_events",
    local_dir="./extracted_txt_files_2025/2025_events",
    cutoff_date="2025-10-31"
):
    """
    Download SWPC event text files newer than cutoff_date (YYYY-MM-DD).
    Example file name pattern: 20251031events.txt
    """
    os.makedirs(local_dir, exist_ok=True)
    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")

    ftp = ftplib.FTP(ftp_server)
    ftp.login()
    ftp.cwd(ftp_path)

    files = ftp.nlst()
    pattern = re.compile(r"(\d{8})events\.txt", re.IGNORECASE)

    downloaded = []
    for file in files:
        m = pattern.match(file)
        if not m:
            continue

        file_date = datetime.strptime(m.group(1), "%Y%m%d")
        if file_date <= cutoff_dt:
            continue  # skip older files

        local_file = os.path.join(local_dir, file)
        if os.path.exists(local_file):
            print(f"Skip existing: {file}")
            continue

        print(f"Downloading {file} ...")
        with open(local_file, "wb") as f:
            ftp.retrbinary(f"RETR {file}", f.write)
        downloaded.append(file)

    ftp.quit()
    print(f"Downloaded {len(downloaded)} new files.")
    return True

def list_swpc_txt_files(txt_root, year=2025, cutoff_yyyymmdd="20251031"):
    """
    Return a list of txt filenames (not paths) under `txt_root` to parse.
    Keeps only files with names like YYYYMMDDevents.txt and later than cutoff.
    """
    txt_root = Path(txt_root)
    patt = re.compile(r"(\d{8})events\.txt$", re.IGNORECASE)

    keep = []
    for f in sorted(txt_root.glob("*.txt")):
        if f.name.lower() == "readme.txt":
            continue
        m = patt.match(f.name)
        if not m:
            continue
        if m.group(1) > cutoff_yyyymmdd:
            keep.append(f.name)
    return keep

def extract_xrs_to_csv(txt_files, extract_to_path, year=2025, out_csv="ftp_flares_2025.csv"):
    """
    Read the given daily 'events' txt files and extract GOES/XRS (flare) rows.
    Overwrites `out_csv` on each run.

    Parameters
    ----------
    txt_files : list[str]
        Filenames (not paths) returned by `list_swpc_txt_files(...)`.
    extract_to_path : str or Path
        Parent folder containing the daily txt files.
        For year 2025 we expect files in {extract_to_path}/2025_events/.
    year : int
        Year of these daily files (used only for building the subfolder name).
    out_csv : str
        Output CSV path; will be overwritten.
    """
    extract_to_path = Path(extract_to_path)
    start_time, peak_time, end_time = [], [], []
    cls, noaa_ar, obs = [], [], []

    if not txt_files:
        # nothing to parse; write empty CSV with headers for reproducibility
        pd.DataFrame(columns=["start_time","peak_time","end_time","class","noaa_ar","obs"]).to_csv(out_csv, index=False)
        print(f"Saved (empty) {out_csv}")
        return out_csv

    for txt_file in txt_files:
        if txt_file.lower() == "readme.txt":
            continue

        # daily file lives in {extract_to_path}/2025_events/ or just extract_to_path/
        if year == 2025:
            path = extract_to_path / "2025_events" / txt_file
        else:
            path = extract_to_path / txt_file

        if not path.exists():
            # if the files were saved directly under extract_to_path, try that
            alt = extract_to_path / txt_file
            if alt.exists():
                path = alt
            else:
                print(f"Skip missing: {path}")
                continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if len(lines) < 14:
            continue

        raw_date = lines[2].strip()
        # Extract YYYY.MM.DD (or variants) and normalize to YYYY/MM/DD
        date_part = (
            raw_date.replace("Event Date:", "")
                    .replace("-", ".").replace("/", ".").replace(" ", ".")
                    .strip(".")
        )
        parts = [p for p in date_part.split(".") if p.isdigit()]
        if len(parts) >= 3:
            date_str = f"{parts[0]}/{parts[1]}/{parts[2]}"  # YYYY/MM/DD
        else:
            # fallback: skip file if date header is not as expected
            continue

        # If the table says "NO EVENT REPORTS." then skip
        rest = lines[13:]
        if not rest or rest[0].strip().upper().startswith("NO EVENT REPORTS"):
            continue

        for line in rest:
            if not line.strip():
                continue
            # remove '+' signs, split on whitespace
            tokens = line.replace("+", "").split()
            if len(tokens) < 9:
                continue

            # tokens[4] should be obs, and we only keep GOES rows (starting with 'G')
            if not tokens[4].startswith("G"):
                continue

            # times are HHMM in tokens[1], [2], [3]
            try:
                s_hhmm = tokens[1]; p_hhmm = tokens[2]; e_hhmm = tokens[3]
                s = f"{date_str} {s_hhmm[:2]}:{s_hhmm[2:]:0>2}"
                p = f"{date_str} {p_hhmm[:2]}:{p_hhmm[2:]:0>2}"
                e = f"{date_str} {e_hhmm[:2]}:{e_hhmm[2:]:0>2}"
            except Exception:
                continue

            start_time.append(s)
            peak_time.append(p)
            end_time.append(e)

            # class at tokens[8]
            cls.append(tokens[8])
            obs.append(tokens[4])

            # NOAA AR present when len == 11 → last token; else 0
            ar = tokens[-1] if len(tokens) == 11 and tokens[-1].isdigit() else "0"
            if ar == "0":
                noaa_ar.append("0")
            else:
                noaa_ar.append('1'+ar)

    df = pd.DataFrame({
        "start_time": start_time,
        "peak_time":  peak_time,
        "end_time":   end_time,
        "class":      cls,
        "noaa_ar":    noaa_ar,
        "obs":        obs
    })

    # sort by start_time 
    # df["start_time"] = pd.to_datetime(df["start_time"])
    # df = df.sort_values("start_time").reset_index(drop=True)

    # Overwrite output CSV on each run
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(df)} XRS flare rows.")
    return out_csv