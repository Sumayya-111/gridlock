# Helmet Detection Traffic Models

YOLO models for real-time traffic violation monitoring (motorcyclist → helmet/plate → OCR).

## Stages

| Stage | Files | Purpose |
|-------|-------|---------|
| `stage1/` | detect_vehicle.pt/onnx | Motorcyclist detection |
| `stage2/` | mu_bien_so_stage2.pt/onnx | Helmet / no-helmet / license plate |
| `stage3/` | ocr_plate.pt/onnx | License plate character OCR |

## Download

```bash
pip install huggingface_hub
huggingface-cli download 2vhoc/helmet-detection-traffic --local-dir ./models
```

Or from the app repo:

```bash
bash scripts/download_models.sh
```
