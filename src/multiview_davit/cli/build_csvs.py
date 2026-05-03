"""CLI: Build CBIS-DDSM CSV metadata from raw DICOM + description files."""

import argparse
from pathlib import Path

from ..data.csv_builder import build_cbis_full_csv


def main():
    parser = argparse.ArgumentParser(description="Build CBIS-DDSM CSV metadata")
    parser.add_argument("--cbis-raw", required=True, help="Path to raw CBIS-DDSM directory")
    parser.add_argument("--cbis-processed", required=True, help="Path to preprocessed JPEG directory")
    parser.add_argument("--out", required=True, help="Output directory for CSV files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building CBIS-DDSM CSV (mass + calc, train + test)...")
    df = build_cbis_full_csv(args.cbis_raw, args.cbis_processed)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_df.to_csv(out_dir / "cbis_train_full.csv", index=False)
    test_df.to_csv(out_dir / "cbis_test.csv", index=False)

    print(f"Train: {len(train_df)} rows → {out_dir / 'cbis_train_full.csv'}")
    print(f"Test:  {len(test_df)} rows → {out_dir / 'cbis_test.csv'}")


if __name__ == "__main__":
    main()
