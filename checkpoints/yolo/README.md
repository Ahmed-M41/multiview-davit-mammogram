# YOLOv5 Breast ROI Detector

This directory holds `best.pt` — the YOLOv5 model trained to detect the breast region in raw mammogram DICOMs.

## Download

```bash
bash scripts/download_yolo_weights.sh
```

The weights are archived at Zenodo (DOI: [ZENODO_DOI]).

## Usage

The detector is invoked automatically by the preprocessing pipeline:

```bash
python -m multiview_davit.cli.preprocess \
  --src data/raw/cbis_ddsm \
  --dst data/processed/cbis_ddsm \
  --weights checkpoints/yolo/best.pt \
  --yolov5-path /path/to/yolov5
```

## YOLOv5 Repository Requirement

Detection requires the [YOLOv5 repository](https://github.com/ultralytics/yolov5) to be available. Clone it and set:

```bash
export YOLOV5_PATH=/path/to/yolov5
```

or pass `--yolov5-path /path/to/yolov5` to the CLI.
