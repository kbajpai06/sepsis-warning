"""Leakage-safe feature engineering. Every feature at hour t uses data from hours <= t only."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DYNAMIC, LABEL, PID, RANGES, STATIC, TIME, WINDOWS


def clean_ranges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in RANGES.items():
        if col in df:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Range-clean, forward-fill (last observation carried forward) per patient, and
    record measurement indicators + hours since last real measurement."""
    df = df.sort_values([PID, TIME]).reset_index(drop=True)
    df = clean_ranges(df)
    for c in DYNAMIC:
        if c not in df:
            df[c] = np.nan
    measured = df[DYNAMIC].notna()
    filled = df.groupby(PID, sort=False)[DYNAMIC].ffill()

    keep = [PID, TIME] + STATIC + ([LABEL] if LABEL in df else [])
    parts = {}
    for c in DYNAMIC:
        parts[c] = filled[c]
        parts[f"{c}_measured"] = measured[c].astype("int8")
        last_t = df[TIME].where(measured[c]).groupby(df[PID]).ffill()
        parts[f"{c}_age"] = (df[TIME] - last_t).clip(upper=48)
    return pd.concat([df[keep], pd.DataFrame(parts, index=df.index)], axis=1)


def add_temporal_features(pre: pd.DataFrame) -> pd.DataFrame:
    """Trailing rolling mean / std / slope over 3, 6, 12 h + shock index."""
    g = pre.groupby(PID, sort=False)
    cols = {}
    for c in DYNAMIC:
        gc = g[c]
        for w in WINDOWS:
            cols[f"{c}_mean{w}"] = gc.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            cols[f"{c}_std{w}"] = gc.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            cols[f"{c}_slope{w}"] = (pre[c] - gc.shift(w)) / w          # units per hour
    feats = pd.concat([pre, pd.DataFrame(cols, index=pre.index)], axis=1)
    feats["shock_index"] = feats["HR"] / feats["SBP"]
    feats["shock_index_mean6"] = (
        feats.groupby(PID, sort=False)["shock_index"].rolling(6, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns
            if c not in (PID, LABEL) and not c.endswith("_measured")]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_temporal_features(preprocess(df))


def truncate(df: pd.DataFrame, max_hours: int) -> pd.DataFrame:
    return df[df[TIME] <= max_hours].reset_index(drop=True)


def patient_split(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=42):
    """Split by PATIENT (never by row) and stratify on ever-septic."""
    ever = df.groupby(PID)[LABEL].max()
    ids, y = ever.index.to_numpy(), ever.to_numpy()
    tr, tmp, _, y_tmp = train_test_split(ids, y, test_size=val_size + test_size,
                                         stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=test_size / (val_size + test_size),
                              stratify=y_tmp, random_state=seed)
    return set(tr), set(va), set(te)
