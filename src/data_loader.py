"""Auto-discovery and loading of the raw CIC-IDS2017 CSV files."""
import glob
import os
import pandas as pd

from config import DATA_RAW, LEAKAGE_COLUMNS


def find_dataset_directory():
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    for root, dirs, files in os.walk(project):
        if ".git" in root:
            continue
        for d in dirs:
            if "IDS2017" in d:
                candidates.append(os.path.join(root, d))
    if candidates:
        return candidates[0]
    if os.path.isdir(DATA_RAW):
        return DATA_RAW
    return None


def list_raw_files(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = find_dataset_directory()
    if dataset_dir is None:
        raise FileNotFoundError(
            "No CIC-IDS2017 CSV directory found. Put the CSV files in "
            "data/raw/ or in a folder whose name contains 'IDS2017'.")
    return sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))


def detect_encoding(path):
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as fh:
                fh.read(200000)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"


def load_raw_file(path):
    enc = detect_encoding(path)
    df = pd.read_csv(path, encoding=enc, low_memory=False)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    seen = set()
    keep = []
    for c in df.columns:
        if c not in seen:
            keep.append(c)
            seen.add(c)
    df = df.loc[:, keep]
    return df


def load_all_raw(dataset_dir=None):
    files = list_raw_files(dataset_dir)
    frames = []
    for f in files:
        name = os.path.basename(f).split(".")[0]
        frames.append((name, load_raw_file(f)))
    return frames