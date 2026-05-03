"""Patient-level stratified train/validation splitting.

Uses GroupShuffleSplit to guarantee zero patient overlap between splits.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_patient_stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    group_col: str = "patient_id",
    label_col: str = "cancer",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into train/val with no patient appearing in both sets."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    groups = df[group_col].values
    train_idx, val_idx = next(gss.split(df, df[label_col], groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def verify_no_patient_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_col: str = "patient_id",
) -> None:
    """Raise AssertionError if any patient appears in both splits."""
    overlap = set(train_df[group_col]) & set(val_df[group_col])
    assert len(overlap) == 0, f"Patient leakage detected — shared patients: {overlap}"
