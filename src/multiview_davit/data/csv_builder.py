"""CSV builder utilities for CBIS-DDSM dataset.

Reads both mass and calc description CSVs, deduplicates on (patient_id, breast, view),
and adds binary label column. Per paper §3.1, both masses and calcifications are included.
"""

import glob
import hashlib
import os
from pathlib import Path

import pandas as pd


def _binary_from_pathology(pathology: str) -> int:
    return 1 if str(pathology).upper() == "MALIGNANT" else 0


def update_dataframe_with_paths(df: pd.DataFrame, processed_dir: str | Path) -> pd.DataFrame:
    """Map each row's image-file-path folder identifier to an actual JPEG under processed_dir."""
    processed_dir = str(processed_dir)
    jpeg_paths = glob.glob(os.path.join(processed_dir, "**", "*.jp*g"), recursive=True)
    jpeg_paths = [p.replace(os.sep, "/") for p in jpeg_paths]

    folder_to_path: dict[str, str] = {}
    for p in jpeg_paths:
        folder_id = os.path.basename(os.path.dirname(p))
        if folder_id and folder_id != ".":
            folder_to_path[folder_id] = p

    out = df.copy()
    lookup_keys = out["image file path"].astype(str).apply(
        lambda x: x.split("/")[-2] if pd.notna(x) and len(x.split("/")) >= 2 else None
    )
    out["image_path"] = lookup_keys.map(folder_to_path)
    out["cancer"] = out["pathology"].apply(_binary_from_pathology)
    out.drop(columns=["cropped image file path", "ROI mask file path"], inplace=True, errors="ignore")
    return out


def build_cbis_full_csv(raw_dir: str | Path, processed_dir: str | Path) -> pd.DataFrame:
    """Build a unified CBIS-DDSM DataFrame from both mass and calc description CSVs.

    Returns a DataFrame with columns including patient_id, breast, view, image_path, cancer.
    """
    raw_dir = Path(raw_dir)
    mass_train = list(raw_dir.glob("**/mass_case_description_train_set.csv"))
    mass_test = list(raw_dir.glob("**/mass_case_description_test_set.csv"))
    calc_train = list(raw_dir.glob("**/calc_case_description_train_set.csv"))
    calc_test = list(raw_dir.glob("**/calc_case_description_test_set.csv"))

    dfs = []
    for csv_path in mass_train + mass_test + calc_train + calc_test:
        df = pd.read_csv(csv_path)
        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]
        split = "test" if "test" in str(csv_path).lower() else "train"
        lesion = "calc" if "calc" in str(csv_path).lower() else "mass"
        df["split"] = split
        df["lesion_type"] = lesion
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CBIS-DDSM description CSVs found under {raw_dir}")

    combined = pd.concat(dfs, ignore_index=True)

    # Normalise key column names across mass/calc schema differences
    combined = combined.rename(columns={
        "left or right breast": "breast",
        "image view": "view",
        "patient_id": "patient_id",
        "pathology": "pathology",
    })

    dedup_keys = ["patient_id", "breast", "view"]
    if "pathology" in combined.columns:
        combined["_malig"] = (combined["pathology"].str.upper() == "MALIGNANT").astype(int)
        combined = (
            combined
            .sort_values(dedup_keys + ["_malig"], ascending=[True, True, True, False])
            .drop_duplicates(subset=dedup_keys + ["split"], keep="first")
            .drop(columns="_malig")
        )

    combined = update_dataframe_with_paths(combined, processed_dir)
    return combined
