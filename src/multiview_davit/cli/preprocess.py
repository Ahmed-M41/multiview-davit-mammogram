"""CLI: Run the full preprocessing pipeline (YOLO ROI crop → threshold → resize → save)."""

import argparse

from ..preprocessing.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Preprocess mammogram images")
    parser.add_argument("--src", required=True, help="Source directory (raw DICOMs or JPEGs)")
    parser.add_argument("--dst", required=True, help="Destination directory for preprocessed JPEGs")
    parser.add_argument("--weights", default="checkpoints/yolo/best.pt", help="YOLOv5 weights path")
    parser.add_argument("--out-size", type=int, default=512, help="Output image size (default 512)")
    parser.add_argument("--threshold", type=int, default=20, help="Background threshold value")
    parser.add_argument("--yolov5-path", default=None, help="Path to YOLOv5 repo (or set YOLOV5_PATH env var)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    results = run_pipeline(
        src_dir=args.src,
        dst_dir=args.dst,
        weights_path=args.weights,
        out_size=args.out_size,
        threshold_value=args.threshold,
        yolov5_path=args.yolov5_path,
        max_workers=args.workers,
    )

    total = sum(len(v) for v in results.values())
    print(f"\nDone — {total} images processed")
    print(f"  ok:         {len(results['ok'])}")
    print(f"  missing:    {len(results['missing'])}")
    print(f"  read_fail:  {len(results['read_fail'])}")
    print(f"  write_fail: {len(results['write_fail'])}")


if __name__ == "__main__":
    main()
