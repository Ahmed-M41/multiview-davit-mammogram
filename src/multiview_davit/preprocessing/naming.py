"""Utilities for mapping source paths to destination JPEG paths."""

import hashlib
import os
from pathlib import Path


def dst_from_src(src: str | Path, out_dir: str | Path) -> str:
    """Map a source image path to a collision-free JPEG path under out_dir.

    Uses basename + 10-char MD5 of the full source path to avoid name collisions.
    """
    src = str(src)
    out_dir = str(out_dir)
    base = os.path.splitext(os.path.basename(src))[0]
    h = hashlib.md5(src.encode("utf-8")).hexdigest()[:10]
    return os.path.join(out_dir, f"{base}-{h}.jpg")
