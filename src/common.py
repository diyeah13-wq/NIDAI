"""Shared experiment plumbing: split loading and the rare-aware training sample.

Every experiment must use the SAME training budget and the SAME held-out test
set so model comparisons are fair:
  - train on MODEL_SAMPLE_SIZE rows (all rare attack rows kept, rest stratified)
  - evaluate on the FULL test set
"""
import numpy as np
import pandas as pd

from config import (PROCESSED_TRAIN, PROCESSED_TEST, FEATURE_ORDER, CLASS_ORDER,
                    RANDOM_STATE, MODEL_SAMPLE_SIZE, MODEL_RARE_CLASSES)


def load_split(path):
    """Load a processed split, drop UNKNOWN labels, return X, y, n_rows.

    y is integer-encoded per CLASS_ORDER (same encoding everywhere). For a
    range in features a model filters X by column index - never by memory.
    """
    df = pd.read_csv(path, usecols=FEATURE_ORDER + ["LabelGroup"],
                     dtype={c: "float32" for c in FEATURE_ORDER})
    df = df[df["LabelGroup"] != "UNKNOWN"]
    y = df.pop("LabelGroup").map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    return df[FEATURE_ORDER].to_numpy(np.float32), y, len(df)


def load_train():
    return load_split(PROCESSED_TRAIN)


def load_test():
    return load_split(PROCESSED_TEST)


def rare_aware_sample(df, n=MODEL_SAMPLE_SIZE):
    """Stratified n-row sample that keeps EVERY instance of the rare classes.

    df must have a 'LabelGroup' column holding integer codes
    (0..K-1 per CLASS_ORDER). Rarities are so extreme (<0.1% of rows) that a
    plain random sample would contain almost zero WebAttack/Botnet examples.
    """
    rare_codes = [CLASS_ORDER.index(c) for c in MODEL_RARE_CLASSES]
    rare = df[df["LabelGroup"].isin(rare_codes)]
    rest = df[~df["LabelGroup"].isin(rare_codes)]
    take = n - len(rare)
    if take < len(rest):
        n_rest = len(rest)
        parts = []
        for g, sub in rest.groupby("LabelGroup"):
            parts.append(sub.sample(int(round(take * len(sub) / n_rest)),
                                    random_state=RANDOM_STATE))
        rest = pd.concat(parts, ignore_index=True)
    sample = pd.concat([rare, rest]).sample(frac=1.0, random_state=RANDOM_STATE)
    return sample