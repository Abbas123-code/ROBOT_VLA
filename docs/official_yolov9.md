# Official YOLOv9-S: perception working; bottle placement blocked

## Strict source selection

Only [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) is used, pinned to
commit `5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff`. The upstream checkout is
unmodified. No Ultralytics package, model wrapper or weights are used.

The authors' official **yolov9-s-converted.pt** release is the inference-ready
checkpoint, not an Ultralytics conversion. SHA256:

`09bf9ca4adef37944f4406455b5b81b451c1e67307916e4702da68cda4d3e46d`

[Provenance configuration](../configs/yolov9_official.json) records the URL,
revision and checksum. The adapter rejects a changed checkpoint or modified
tracked upstream files before deserialization. Only this pinned trusted pickle
is loaded with `weights_only=False`; do not substitute untrusted checkpoints.
The upstream code is **GPL-3.0**, not the ROS package's Apache-2.0 license.
Review the upstream license before redistribution or product integration.

## Separate CPU environment

Inference uses **env_yolov9_cpu**, Python 3.12, PyTorch **2.5.1+cpu** and
torchvision **0.20.1+cpu**. [Pinned dependencies](../configs/requirements-yolov9-cpu.txt)
describe the inference-only dependency set. No NVIDIA packages are needed there.

The existing Isaac environment's torch 2.11.0+cu128 / torchvision 0.26.0 failed
to import `torchvision::nms`. They were not replaced. Initially, CPU-side helper
packages (pandas, seaborn, tqdm, IPython and GitPython plus dependencies) were
installed there before the separate environment was created. No driver, system
CUDA, GPU setting, Isaac torch or torchvision changes were made.

## Recognition and localization

[The adapter](../src/perception/yolov9_official.py) runs the official model,
letterboxing and NMS on CPU. The bottle is small in the 640×480 wrist image:
full-frame inference missed it. Uniform 320×320 overlapping tiles (50% overlap),
each processed at 640 pixels, detected it. Class-aware NMS merges duplicates.
The tile grid does not use a known bottle location or colour detector to choose
the crop. Confidence remains **0.50**; the threshold was not lowered to force a
successful result.

The official repository's reference image produced five horse detections. The
recorded table image produced a bottle score of approximately **0.59**. Five
fresh ROS observations produced bottle scores around **0.62–0.68** in one live
test and a stable world XY estimate around **(0.72245, 0.34204) m**.

**False positives also occur**, including umbrella/refrigerator labels on other
parts of this synthetic scene. Reports retain them. This is not reliable general
recognition of every table object. COCO's standard classes also do not directly
provide a generic receiving-box task target.

Only one confident `bottle` detection is accepted. Within its bounding box,
registered depth, camera calibration and exact-time TF feed the existing cyan
body/known-radius cylinder fit. Five stable frames are required. Thus this is
**YOLO recognition plus model-specific RGB-D localization**, not YOLO-only 3D
pose estimation. If YOLO finds no bottle, there is no colour-only fallback.
The blue box remains a known fixture with checked coordinates, not a YOLO target.

Start **Panda: wrist RGB-D camera (no motion)** in VS Code, then:

```bash
source /opt/ros/jazzy/setup.bash
./env_yolov9_cpu/bin/python scripts/detect_bottle_rgbd.py --detector official-yolov9
```

For a recorded-image test only:

```bash
./env_yolov9_cpu/bin/python scripts/check_official_yolov9.py
```

- [Live detection report](../ros2_ws/log/bottle_yolov9_detection.json)
- [Live annotated image](../ros2_ws/log/bottle_yolov9_detection.png)
- [Recorded-image report](../ros2_ws/log/yolov9_official_validation.json)
- [Recorded-image detections including false positives](../ros2_ws/log/yolov9_official_detection.png)

## Robot trial results — not ready for visible execution

The user reauthorized bottle pickup and placement using this official model.
A separate domain-46 controller used fresh domain-44 YOLO/RGB-D snapshots.
Physics ground truth only rejected mismatches and checked contacts, slip, tilt,
neighbour movement and outcomes. It did not correct the image target.

Three headless trials were performed:

1. A 175 mm neck-grasp target triggered palm/cap contact during descent.
2. A 195 mm upper-neck/rigid-cap target avoided that contact and achieved
   bilateral finger contact, but exceeded the **4 mm slip guard** during lift.
3. Increasing the grip limit from **5 N to 8 N per finger**, with a stronger
   closure target, still triggered the guard at approximately **4.019 mm**.

No bottle transfer or release was reached. No visible execution was launched.
The cap is part of the single rigid procedural bottle, not a removable real cap.
No object mass, friction, collision geometry, teleportation or attachment was
changed to make the grasp pass. The 4 mm guard was not weakened.

- [Failed isolated validation report](../ros2_ws/log/bottle_yolo_place_validation.json)
- [Experimental placement controller](../isaac/scripts/yolo_bottle_place_controller.py)
- [Fingerprint-gated launcher](../isaac/scripts/run_bottle_pick.py)

The launcher's `--yolo-place` mode requires a successful matching validation
before `--execute`. The current report is **completed: false**, so visible
execution remains blocked. No motion task was added to VS Code. A redesigned
grasp requires a new isolated validation, not bypassing these checks.