"""BI-RADS to binary label mapping for the INbreast dataset.

Per paper §3.1 and the dataset codebook:
  BI-RADS 1, 2, 3  → 0  (Benign)
  BI-RADS 4a, 4b, 4c, 5, 6 → 1  (Malignant)
"""

import re

# Canonical mapping from BI-RADS code string to binary class
BIRADS_TO_BINARY: dict[str, int] = {
    "1": 0,
    "2": 0,
    "3": 0,
    "4a": 1,
    "4b": 1,
    "4c": 1,
    "4A": 1,
    "4B": 1,
    "4C": 1,
    "5": 1,
    "6": 1,
}

# Numeric mapping (for CSV columns that store integer BI-RADS values)
_NUMERIC_TO_BINARY: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}


def birads_to_binary(code: str | int | float) -> int:
    """Convert a BI-RADS code to binary label (0=Benign, 1=Malignant).

    Handles string codes like '4a', '4B', '5', and numeric codes like 4, 5.
    """
    if isinstance(code, float):
        code = int(code)
    if isinstance(code, int):
        return _NUMERIC_TO_BINARY[code]
    code = str(code).strip()
    if code in BIRADS_TO_BINARY:
        return BIRADS_TO_BINARY[code]
    # Try matching numeric prefix for codes like "4a", "4b", "4c"
    m = re.match(r"^(\d+)([a-cA-C]?)$", code)
    if m:
        num = int(m.group(1))
        sub = m.group(2).lower()
        key = f"{num}{sub}" if sub else str(num)
        if key in BIRADS_TO_BINARY:
            return BIRADS_TO_BINARY[key]
        return _NUMERIC_TO_BINARY.get(num, -1)
    raise ValueError(f"Unknown BI-RADS code: {code!r}")
