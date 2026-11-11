# Official YOLOv9-S tabletop fine-tuning

This workflow uses **WongKinYiu/yolov9**, pinned to
`5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff`, with `train_dual.py` and
`models/detect/yolov9-s.yaml`. It does not use the Ultralytics package.
It performs image training only: no robot commands or simulation motion.

## Dataset

The uploaded CVAT export contains 73 PNG images and 423 COCO bounding boxes in
`data/datasets/images/` and `data/datasets/labels/instances_default.json`.
The original images and annotations are preserved.

`scripts/prepare_yolov9_dataset.py` validates dimensions, bounds, category and
image IDs, duplicates, and unsupported crowd/rotated labels. It writes a new
YOLO-format copy under `data/datasets/yolov9_tabletop_v1/` and refuses overwrite.

Two uploaded views from the prior quality review are excluded from the derived
copy: `zoom_table_a180_wide.png` and `zoom_table_a180_wider.png`. The result is:

| Split | Images | Bounding boxes |
|---|---:|---:|
| Train | 58 | 331 |
| Validation | 13 | 91 |

The split is deterministic, grouped by 60-degree camera azimuth sector. Related
near/medium/far, wide/wider and orbit-elevation views stay together. All seven
classes occur in both splits. Audit hashes, excluded views, split membership and
class counts are in `audit.json`; `annotation_preview.jpg` shows sample labels.

| YOLO ID | COCO ID | Name |
|---:|---:|---|
| 0 | 1 | Blue Tray |
| 1 | 2 | Red Block |
| 2 | 3 | Black Block |
| 3 | 4 | Green Block |
| 4 | 5 | Yellow Bar |
| 5 | 6 | Orange Can |
| 6 | 7 | Water Bottle |

**These are custom IDs, not COCO's pretrained class IDs.** The original live
COCO detector and the separate custom seven-class adapter have different contracts.
Use the [trained wrist detector](trained_wrist_detection.md); do not replace the
original detector's checkpoint blindly.

## Isolated environment and official weights

Use `env_yolov9_train`, separate from Isaac Sim and CPU inference. The selected
pair is torch 2.7.1+cu128 / torchvision 0.22.1+cu128 from PyTorch's official cu128
index, with `configs/requirements-yolov9-train.txt`. CUDA runtime libraries are
environment-local packages; this does not replace drivers or system CUDA.

Original dual-head pretrained weights:
<https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s.pt>

Downloaded SHA-256:
`e232b9c0423ff04608d973e684f8ee808f933e6b7fc052787efb30e43b8acac5`

This is distinct from yesterday's converted inference checkpoint. The launcher
checks the original training checkpoint hash and tracked upstream changes before
loading. Upstream full-model checkpoints require legacy pickle loading with
PyTorch 2.6+: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is scoped to this process.
Only use trusted official weights and locally generated training checkpoints.
The launcher also provides a process-local Pillow `getsize` compatibility shim
for upstream annotation plots; the official checkout remains unmodified.

## Run

From the project root, in the prepared environment:

```bash
./env_yolov9_train/bin/python tests/test_yolov9_dataset.py
./env_yolov9_train/bin/python scripts/prepare_yolov9_dataset.py
./env_yolov9_train/bin/python scripts/train_official_yolov9.py
```

Skip preparation when the derived dataset already exists. Defaults: maximum
100 epochs, batch 8, 640-pixel input, AdamW, cosine learning rate, seed 42,
early-stopping patience 25. Hue changes and mixup are disabled because colour
defines several classes. Mild augmentation is in `configs/yolov9_tabletop_hyp.yaml`.

Run names must be new; previous outputs are never overwritten. Outputs live in
`outputs/training/<name>/`: best/last weights, metrics CSV, plots, TensorBoard
events, provenance and a completion summary. Training datasets, environments and
checkpoints remain ignored by Git. No cloud experiment tracker is installed.

## Completed run: 2026-10-21

`tabletop_yolov9s_v1` completed 100 epochs in approximately 5.8 minutes.
The best checkpoint is `outputs/training/tabletop_yolov9s_v1/weights/best.pt`
(20.3 MB), SHA-256:
`28db4ee45eb19eb6d44d7cb9bcc8d0c13b8a4fda95f57dcd89dd508ef9910841`.
Its final upstream revalidation on 13 images / 91 objects reported:

| Metric | All classes | Water Bottle |
|---|---:|---:|
| Precision | 0.962 | 0.996 |
| Recall | 0.976 | 1.000 |
| AP@50 (mean for all classes) | 0.966 | 0.995 |
| AP@50:95 (mean for all classes) | 0.818 | 0.767 |

Black Block was the weakest class (AP@50 0.790, AP@50:95 0.617).
The prediction montage also shows a false Orange Can detection on the robot.
The selected training CSV epoch is zero-based 85; its metrics differ slightly
from the final fused-checkpoint revalidation above. Upstream records zero
validation losses here because its loss calculation is disabled; those zeros
are not evidence of perfect predictions. TensorBoard's optional model graph
could not be traced, but scalar logging, plots, training and validation completed.
No training exceptions were found. Original annotation hash and tracked upstream
source remained unchanged. The model was subsequently integrated through a
separate [trained wrist detector](trained_wrist_detection.md), preserving the
original pretrained detector.

## Evaluation limits

This is a small dataset from **one synthetic scene**. Grouped validation reduces
near-duplicate leakage but is not an independent scene or real-world test.
Validation is used for checkpoint selection; there is no separate test set.
Collect new object layouts, lighting and scenes before claiming generalisation.
No score authorises robot movement or bypasses the existing motion guards.