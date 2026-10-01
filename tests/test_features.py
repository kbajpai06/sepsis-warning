import numpy as np
import pandas as pd

from sepsis.config import DYNAMIC, LABEL, PID, TIME
from sepsis.data.synthetic import make_synthetic
from sepsis.features import build_features, feature_columns, patient_split


def test_no_future_leakage():
    """Features at hour t must not change if data after t changes."""
    df = make_synthetic(20, seed=1)
    pid = df[PID].iloc[0]
    t0 = 20
    base = build_features(df)
    mod = df.copy()
    future = (mod[PID] == pid) & (mod[TIME] > t0)
    mod.loc[future, DYNAMIC] = mod.loc[future, DYNAMIC] * 3 + 5
    new = build_features(mod)
    cols = feature_columns(base)
    a = base[(base[PID] == pid) & (base[TIME] <= t0)][cols].reset_index(drop=True)
    b = new[(new[PID] == pid) & (new[TIME] <= t0)][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_forward_fill_and_staleness():
    df = pd.DataFrame({PID: [1] * 4, TIME: [1, 2, 3, 4], "HR": [80, np.nan, np.nan, 90],
                       "Lactate": [2.0, np.nan, np.nan, np.nan], "Age": 50.0, "Gender": 1.0, LABEL: 0})
    f = build_features(df)
    assert f["Lactate"].tolist() == [2.0] * 4
    assert f["Lactate_age"].tolist() == [0, 1, 2, 3]
    assert f["HR_age"].tolist() == [0, 1, 2, 0]


def test_rolling_slope_is_trailing():
    df = pd.DataFrame({PID: [1] * 8, TIME: range(1, 9), "HR": [60, 62, 64, 66, 68, 70, 72, 74],
                       "Age": 50.0, "Gender": 1.0, LABEL: 0})
    f = build_features(df)
    assert np.isnan(f.loc[2, "HR_slope3"])               # not enough history yet
    assert np.isclose(f.loc[7, "HR_slope3"], 2.0)        # +2 bpm/hour


def test_out_of_range_values_removed():
    df = pd.DataFrame({PID: [1, 1], TIME: [1, 2], "HR": [500, 80], "Age": 50.0, "Gender": 1.0, LABEL: 0})
    assert np.isnan(build_features(df).loc[0, "HR"])


def test_patient_split_has_no_overlap():
    df = make_synthetic(300, seed=3)
    tr, va, te = patient_split(df)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == df[PID].nunique()
