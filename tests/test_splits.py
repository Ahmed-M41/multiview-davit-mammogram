"""Test that patient-level splitting produces zero patient overlap."""

import pandas as pd
import pytest

from multiview_davit.data.splits import make_patient_stratified_split, verify_no_patient_leakage


def _make_df(n_patients: int = 50, images_per_patient: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_patients):
        label = i % 2
        for j in range(images_per_patient):
            rows.append({
                "patient_id": f"P{i:03d}",
                "cancer": label,
                "image_path": f"img_{i}_{j}.jpg",
                "view": "CC" if j == 0 else "MLO",
            })
    return pd.DataFrame(rows)


def test_no_patient_leakage():
    df = _make_df()
    train_df, val_df = make_patient_stratified_split(df, val_size=0.2)
    verify_no_patient_leakage(train_df, val_df)


def test_split_sizes():
    df = _make_df(n_patients=50)
    train_df, val_df = make_patient_stratified_split(df, val_size=0.2)
    total = len(train_df) + len(val_df)
    assert total == len(df)
    assert len(val_df) > 0
    assert len(train_df) > 0


def test_leakage_detected():
    df = _make_df()
    train_df, val_df = make_patient_stratified_split(df, val_size=0.2)
    # Artificially inject leakage
    leaked = train_df.iloc[[0]].copy()
    val_df_bad = pd.concat([val_df, leaked], ignore_index=True)
    with pytest.raises(AssertionError, match="leakage"):
        verify_no_patient_leakage(train_df, val_df_bad)
