"""Re-create train/test splits from the cleaned CSV without re-reading raw data.

The full preprocessing run was interrupted while writing train.csv.gz
(truncated gzip). The cleaned file is intact, so we redo only the
deterministic train_test_split from it, guaranteeing train/test are
consistent with each other and with data/processed/processed_info.json.
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (PROCESSED_CLEAN, PROCESSED_TRAIN, PROCESSED_TEST,
                    FEATURE_ORDER, RANDOM_STATE, TEST_SIZE)

print("Reading cleaned data (features + LabelGroup only)...", flush=True)
df = pd.read_csv(PROCESSED_CLEAN,
                 usecols=FEATURE_ORDER + ["LabelGroup"],
                 dtype={c: "float32" for c in FEATURE_ORDER})
print("Loaded %d rows x %d cols" % df.shape, flush=True)

X = df[FEATURE_ORDER]
y = df["LabelGroup"]
del df

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
del X, y

tr = Xtr.assign(LabelGroup=ytr)
te = Xte.assign(LabelGroup=yte)
del Xtr, Xte, ytr, yte

tr.to_csv(PROCESSED_TRAIN, index=False, compression="gzip")
te.to_csv(PROCESSED_TEST, index=False, compression="gzip")

print("train:", PROCESSED_TRAIN, len(tr))
print("test :", PROCESSED_TEST, len(te))
print("train class mix:")
for k, v in tr["LabelGroup"].value_counts().items():
    print("  %-12s %d" % (k, v))
print("test class mix:")
for k, v in te["LabelGroup"].value_counts().items():
    print("  %-12s %d" % (k, v))
print("DONE")