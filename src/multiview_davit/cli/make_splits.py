"""CLI: Create patient-level train/val split from CBIS-DDSM training CSV."""

import argparse
from pathlib import Path

import pandas as pd

from ..data.splits import make_patient_stratified_split, verify_no_patient_leakage


def main():
    parser = argparse.ArgumentParser(description="Create patient-stratified train/val split")
    parser.add_argument("--csv", required=True, help="Path to cbis_train_full.csv")
    parser.add_argument("--out", required=True, help="Output directory for split CSVs")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    train_df, val_df = make_patient_stratified_split(
        df, val_size=args.val_size, random_state=args.seed
    )
    verify_no_patient_leakage(train_df, val_df)

    train_df.to_csv(out_dir / "cbis_train.csv", index=False)
    val_df.to_csv(out_dir / "cbis_val.csv", index=False)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"Train patients: {train_df['patient_id'].nunique()}")
    print(f"Val patients:   {val_df['patient_id'].nunique()}")
    print("No patient leakage confirmed.")


if __name__ == "__main__":
    main()
