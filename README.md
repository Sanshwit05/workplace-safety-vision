# Computer Vision–Based Workplace Safety Monitoring System

An on-device PPE monitoring pipeline for people, helmets, and safety vests. It uses Ultralytics YOLO for detection, OpenCV for media I/O, a transparent geometry-based association layer, and a simple IoU tracker with temporal persistence so a single bad frame does not immediately create an alert.

## Requirements and setup

The verified machine has an NVIDIA GeForce RTX 4060 with 8 GB VRAM, driver 621.01, and CUDA 13.5. All runtime files are kept inside this project. The commands below use `uv` only:

```powershell
cd D:\workplace-safety-vision
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv venv --python .python\cpython-3.11.15-windows-x86_64-none\python.exe .venv
uv pip install --python .venv\Scripts\python.exe -e . --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
```

The first YOLO model download is `yolov8n.pt` and is stored by Ultralytics under the project-local runtime/cache when commands are run from this project. The code defaults to GPU `device: 0`; use `--device cpu` if necessary.

## Dataset and license

`scripts/download_dataset.py` downloads the public Ultralytics Construction-PPE dataset from [the official dataset page](https://docs.ultralytics.com/datasets/detect/construction-ppe/) into `data/construction-ppe`. The source page describes 1,416 images and an AGPL-3.0 release. Keep the dataset attribution and license obligations if redistributing it. Ultralytics YOLO is also AGPL-3.0; review that license before commercial redistribution.

## Train, validate, evaluate

```powershell
$py = "$PWD\.venv\Scripts\python.exe"
& $py scripts\download_dataset.py
& $py scripts\train.py --epochs 25
& $py scripts\validate.py --split val
& $py scripts\evaluate.py
```

Training writes `runs/ppe`, copies the best checkpoint to `models/best.pt`, and uses the RTX 4060 with a conservative batch size of 8 and 640px images. Evaluation writes precision, recall, mAP@50, mAP@50–95, and a confusion matrix under `reports/`.

## Inference and reports

```powershell
& $py scripts\infer.py path\to\image.jpg
& $py scripts\infer.py path\to\video.mp4
```

Annotated media is written to `outputs/`. Each run also writes a CSV and JSON report under `reports/`, including per-frame track IDs, status, bounding boxes, confidence, and a compliance summary.

## Configuration

Edit `config/default.yaml` to change model image size, epochs, batch, device, confidence, IoU, frame sampling, spatial association windows, tracker matching IoU, missing-frame tolerance, and the number of frames required before a status is trusted. The five statuses are `compliant`, `missing_helmet`, `missing_vest`, `missing_both`, and `uncertain`.

## Tests

```powershell
& $py -m pytest -q
```

Tests cover PPE association, all compliance classes, temporal persistence, summary statistics, and CSV/JSON generation.

## Limitations

The released construction-PPE labels may contain more classes than the three used by this application; the configured class-name aliases must match the dataset/model names. The association stage is intentionally interpretable (PPE centers must fall in configurable regions of a person box), and the tracker is a lightweight single-camera IoU tracker rather than a re-identification system. Occlusion, unusual camera angles, crowded scenes, and poor lighting can cause `uncertain` or incorrect results. This is an assistive monitoring tool, not a substitute for a safety officer or site policy.
