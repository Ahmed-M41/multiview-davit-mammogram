# Downloads YOLOv5 breast ROI detector weights from Zenodo.
# Usage: .\scripts\download_yolo_weights.ps1

$ZenodoDoi = "10.5281/zenodo.20016511"
$ZenodoRecord = $ZenodoDoi -replace "10.5281/zenodo.", ""
$Url = "https://zenodo.org/record/$ZenodoRecord/files/yolo_best.pt"

New-Item -ItemType Directory -Force -Path "checkpoints\yolo" | Out-Null
Write-Host "Downloading YOLOv5 weights..."
Invoke-WebRequest -Uri $Url -OutFile "checkpoints\yolo\best.pt"
Write-Host "Saved to checkpoints\yolo\best.pt"
