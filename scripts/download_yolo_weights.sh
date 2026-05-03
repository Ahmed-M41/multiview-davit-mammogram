#!/usr/bin/env bash
# Downloads YOLOv5 breast ROI detector weights from Zenodo.
# Usage: bash scripts/download_yolo_weights.sh

set -euo pipefail

ZENODO_DOI="[ZENODO_DOI]"
ZENODO_RECORD=$(echo "$ZENODO_DOI" | sed 's|10.5281/zenodo.||')
URL="https://zenodo.org/record/${ZENODO_RECORD}/files/yolo_best.pt"

mkdir -p checkpoints/yolo
echo "Downloading YOLOv5 weights..."
wget -O checkpoints/yolo/best.pt "$URL"
echo "Saved to checkpoints/yolo/best.pt"
