# Checkpoints

Model weights are **not included** in the repository due to file size.

## Download Pre-trained Weights

All model checkpoints (.pth files) and the YOLOv5 breast detector weights are archived at Zenodo:

- DOI: 10.5281/zenodo.20016511

### Download script

```bash
bash scripts/download_checkpoints.sh
```

or on Windows:

```powershell
.\scripts\download_checkpoints.ps1
```

## Expected Files

```
checkpoints/
├── davit_cc_best.pth       # DaViT-CC (CC-view specialist)
├── davit_mlo_best.pth      # DaViT-MLO (MLO-view specialist)
├── davit_all_cc_best.pth   # DaViT-ALL evaluated on CC view
├── davit_all_mlo_best.pth  # DaViT-ALL evaluated on MLO view
└── yolo/
    └── best.pt             # YOLOv5 breast ROI detector
```

## Training Your Own

To train from scratch, follow the instructions in the main README and use:

```bash
python -m multiview_davit.cli.train --config configs/davit_cc.yaml
```
