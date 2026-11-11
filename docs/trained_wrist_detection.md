# Trained wrist detection: scope, geometry, and tests

## Status and limits

**Latest verified result, 2026-10-21:** the visible guarded validation completed
the full bottle pick, box placement, open-gripper release, retreat and HOME
return. The saved report has `completed: true`, `visible: true`,
`mode: validation`, `returned_home: true`, floor contact and open fingers true.
Bottle final XYZ: `[0.44669932, -0.24377392, 0.00799969]` m. Maximum carry slip:
3.089 mm; maximum joint tracking error: 0.013051 rad; measured HOME joint error:
0.003282 rad. The same-scene camera published 2,443 frames. This is one successful
visible simulation validation, not a separate `--execute` run or hardware test.

The earlier apparent hang was an intentional abort during lowering: a
bottle/table event occurred above the table beneath the tray. The placement
policy now ignores only >=2 mm, exactly zero-impulse table proximity; any
near-touch or nonzero impulse still aborts. Tray support requires loaded
near-touch evidence. Open-finger retreat and HOME travel retain bottle position,
tilt, neighbor, contact and joint tracking checks. Fourteen Cartesian segments
plus a small bounded HOME-joint correction were preflighted; thirteen callback
tests and nineteen command/approval tests passed. The final GUI remains paused
to preserve the completed scene. No automatic repeat or reset occurs.

Historical intermediate failures and development limitations are described below;
they do not supersede this completed visible-validation result.

This is **same-scene simulated tabletop perception**, not a general bottle
detector, speech interface, or generally validated pickup system. The trained model
produces boxes; RGB-D geometry refines the bottle and tray targets. Neither a
box nor a passing offline test approves robot motion.

Verified on 2026-10-21: playback-lock regression and five camera frames passed;
live trained-model RGB-D capture passed; RViz subscribed to both raw and
annotated streams. An isolated motion validation **failed safely during
`bottle_descend`**, before closure, lift or placement. The left finger reported
an unexpected bottle-body contact (reported separation +10.89 mm; this is a
contact notification, not proof of penetration). No automatic retry or visible
execution followed. No GPU settings, drivers or system CUDA were changed.

### Revised clearance grasp: isolated test, 2026-10-21

After explicit authorization, the controller now uses the shared
`clearance_body_grasp()` helper: 60° tilt, 20° fixed world yaw, 143 mm height,
and a 5 mm inward radial offset computed only from vision XY. The offset is an
approach geometry choice, not a simulator-pose correction. Nominal lowering
preflight includes this offset. Orientation stays constant during carry.

The updated kinematic test passed all ten 120-step segments, with maximum joint
step 0.0470022 rad. Eighteen parser/approval tests also passed. These results do
not establish physical collision clearance or grasp stability.

One new isolated physics attempt **failed during descent**, before finger closure.
Added solver diagnostics recorded a +10.751154 mm finger/body separation and
impulse `[0.0, -0.0, 0.0]` N·s. This event was a separated, zero-impulse proximity
report, yet the current strict contact policy rejects it. The physical bottle
remained at approximately `[0.72000009, 0.33999994, -0.00000010]` m. No contact
filter, collision offset, force, material, slip or tilt limit was changed. No
automatic retry, release, or visible execution followed. The latest validation
report remains `completed: false`; the new grasp is **experimental, not approved**.

Next work must explicitly distinguish proximity events from physical contact,
with regression tests and separate bilateral-support evidence. Do not simply
ignore all positive separations: speculative solver contacts may exert impulses,
and proximity alone must never count as a successful grip.

### Visible validation and same-scene video

The launcher now accepts `--validate --yolo-place --trained-tabletop
--show-validation`. This is **visible validation**, not bypassed execution.
It still obtains five fresh registered domain-44 RGB-D target observations
before launching a separate domain-46 physics scene. The frozen source must
be running when starting another attempt. `--show-validation` is rejected with
`--execute`; execution continues to require a matching successful validation.

[isaac/scripts/live_motion_camera.py](../isaac/scripts/live_motion_camera.py)
mounts an RGB camera on the actual controller robot. Its completed render frames
are published from guarded controller ticks without extra physics steps. Three
valid frames are required before task motion. The existing external viewport
shows the robot; it is not replaced with the sensor camera.

VS Code tasks:
- **Panda: YOLO moving wrist (domain 46)**
- **Panda: RViz moving wrist (domain 46)**, using
  [configs/wrist_motion.rviz](../configs/wrist_motion.rviz)
- **Panda: visible bottle validation with camera**

The GUI title is **VISIBLE GUARDED VALIDATION (domain 46)**. Motion starts
automatically; do not click Play. On success or failure the GUI remains paused
with a result title and accidental playback blocked. There is no automatic retry
or release after failure. Close this window before another attempt.

The domain-46 raw and YOLO image topic names match domain 44, but are isolated
by ROS domain. `/wrist_camera/source` identifies `motion_validation`, the camera
path, render token and simulation render time. These are **RGB-only diagnostic
frames**, with wall-clock ROS headers: no moving depth, CameraInfo or TF is
published. Do not mix them with domain-44 depth. Box overlays are live; control
still uses the pre-motion RGB-D snapshot, not visual servoing.

Runtime verification passed: three changing raw images, three same-stamp YOLO
images and joint change 0.009647 rad/m observed by the read-only
[tests/check_live_motion_view.py](../tests/check_live_motion_view.py). RViz was
subscribed to both domain-46 image topics. This proves live motion/video delivery,
**not successful pickup or placement**. At this check the visible run was in
`bottle_hover`; the preceding headless run stopped at 4.0368 mm bottle slip.

The contact policy now distinguishes only >=2 mm, exactly zero-impulse reports
on finger/body or finger/shoulder during descent. Palm/neck/other-object guards
remain strict. Bilateral grip requires near-touch loaded evidence, never mere
proximity. Six classifier and eleven real-callback tests passed. The lift target
now matches its planned command-space target to avoid compounding servo sag;
the minimum lift, force, slip and tilt limits are unchanged.

## Model and class contract

[configs/yolov9_tabletop.json](../configs/yolov9_tabletop.json) selects the local
fine-tuned, dual-head YOLOv9-S checkpoint. This is the official
WongKinYiu/yolov9 implementation, not Ultralytics. The adapter in
[src/perception/yolov9_tabletop.py](../src/perception/yolov9_tabletop.py) checks:

- Upstream revision `5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff` and no tracked
  upstream changes.
- Checkpoint SHA-256
  `28db4ee45eb19eb6d44d7cb9bcc8d0c13b8a4fda95f57dcd89dd508ef9910841`.
- Exact class mapping and the `DualDDetect` head.
- No installed Ultralytics package in the inference environment.

The custom classes are:

| ID | Exact label |
| --- | --- |
| 0 | Blue Tray |
| 1 | Red Block |
| 2 | Black Block |
| 3 | Green Block |
| 4 | Yellow Bar |
| 5 | Orange Can |
| 6 | Water Bottle |

These are **not COCO class IDs**. The current configuration uses CPU inference,
640-pixel inference size, confidence threshold `0.5`, and NMS IoU threshold
`0.45`. Input is `uint8` RGB, not BGR. The adapter uses the main inference head
from the dual model and scales boxes back to the original image.

Checkpoint loading uses `weights_only=False`: only use the trusted local
training artifact whose hash is pinned. Hash checking does not make arbitrary
downloaded pickle-based models safe. No downloads or installs are needed by
the tests when the existing CPU environment and artifacts are available.

## Domain 44: raw input versus annotations

[scripts/wrist_yolo_ros.py](../scripts/wrist_yolo_ros.py) defaults to
`ROS_DOMAIN_ID=44` and `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`. It has no motor
publishers. ROS and model initialization happen inside `main()`, so importing
the script to test `rgb_array()` does not start ROS or load weights.

| Topic | Meaning |
| --- | --- |
| `/wrist_camera/color/image_raw` | Original `sensor_msgs/Image`, `rgb8`; detector input |
| `/wrist_camera/depth/image_raw` | Registered depth, `32FC1`, metres along optical Z |
| `/wrist_camera/color/camera_info` | RGB intrinsics and optical frame |
| `/wrist_camera/depth/camera_info` | Depth registration calibration |
| `/wrist_camera/detections/image` | Annotated RGB image with boxes, labels, scores and latency |
| `/wrist_camera/detections/objects` | `std_msgs/String` JSON: image stamp, frame ID, objects, inference seconds and model hash |
| `/tf` | Transform chain `world` → `panda_hand` → `wrist_camera_optical_frame` |

The annotation publisher preserves the input image header; it does not replace
or modify the raw topic. RViz must use domain 44.
[configs/wrist_camera.rviz](../configs/wrist_camera.rviz) provides separate
**Wrist RGB** and **Wrist YOLO boxes** displays for the raw and annotated topics.
An empty annotated display is not evidence that the raw camera is broken.
The YOLO publisher must be running separately. VS Code task **Panda: trained
wrist YOLO boxes (no motion)** starts it; the camera and RViz have separate tasks.

Normal publisher mode only needs fresh RGB. `--capture` additionally requires
exactly matching RGB, depth and both CameraInfo timestamps, and a world TF at
that image timestamp. The four messages must share the optical frame and image
dimensions; the two intrinsic matrices must match. Supported depth is
little-endian `32FC1`. The RGB decoder accepts row padding but rejects other
encodings, big-endian messages and a row stride shorter than `width * 3`.

Capture accepts images less than two seconds old and collects five valid
observations within ten seconds. The maximum Euclidean deviation from the
median of the combined bottle XY, tray XY and floor-height vector must be no
more than 3 mm. A rejected geometry observation clears that sequence. Capture
has a 60-second deadline. It saves its JSON report and diagnostic image under
the ROS log directory. It does **not** publish annotations in capture mode,
avoiding competition with the separate live annotation publisher. This script
does not itself create the older NPZ used by the optional offline test.

## RGB-D geometry and rejection rules

[src/perception/tabletop_rgbd.py](../src/perception/tabletop_rgbd.py) first
requires exactly one `Water Bottle` / class `6` and one `Blue Tray` / class `0`,
each with confidence at least `0.5`. Matching requires both the exact label and
class ID. Two accepted detections remain ambiguous even if their scores differ;
the helper does not silently choose the highest score.

For pixel `(u, v)` and optical-axis depth `d`, camera coordinates are
`((u-cx)*d/fx, (v-cy)*d/fy, d)`. The camera-to-world transform then applies
rotation and translation. Depth is not radial range. Correct registration,
intrinsics, and the transform at capture time are essential.

- **Bottle:** the detector ROI masks RGB before the existing cyan-component,
  known-radius fit in [src/perception/bottle_rgbd.py](../src/perception/bottle_rgbd.py).
  It assumes the upright opaque cyan bottle on a horizontal table at world
  `Z=0`, radius `0.03225 m`, sufficient visible height and cylinder arc, and a
  consistent depth fit. Color is geometric refinement after the YOLO gate,
  not a replacement when recognition fails.
- **Tray:** select blue pixels inside its detection ROI and world heights
  between 2 and 70 mm. Use the largest connected component with at least 80
  points. Its 1st-to-99th-percentile XY extent must be 175–215 mm on each axis,
  matching a known axis-aligned 200 mm exterior.
- **Floor:** require at least 25 low points inside the central region. Median
  height must be within 3 mm of the known 8 mm floor, and the 90th percentile
  of absolute floor residuals must be at most 2 mm.
- **Obstacle:** inspect all visible depth, not just blue pixels, inside the
  central placement square (65 mm half-width). More than five points over
  `floor + 15 mm` reject placement. This is a visible-depth check, not proof
  that occluded space is empty.
- `rim_z=0.06 m` is a **known geometry constant**, not an independently measured
  rim height. Rotated, differently sized, relocated or differently colored
  objects are not established capabilities of this pipeline.

These perception modules do not read USD object poses or substitute fixed XY
coordinates for failed detections. Their checks are restricted geometry checks,
not a general calibration validator or a complete collision model.

## Text command and execution gate

[scripts/robot_tabletop_command.py](../scripts/robot_tabletop_command.py)
implements a full-string English allowlist, not speech recognition. There is
no microphone, listening mode, speech-to-text component, LLM planner or free-form
task execution. The canonical request is
`pick up the bottle and place it in the blue tray`. Limited wording changes,
including `blue track`, are accepted; negation and appended tasks are rejected.
User text is never inserted into the shell program.

Exactly one explicit `--validate` or `--execute` is required:

- `--validate` starts a separate **headless motion simulation**, not a harmless
  syntax check. It never automatically proceeds to visible execution.
- `--execute` requires a successful matching isolated validation. It does not
  automatically validate, retry, or release on failure.

The CLI selects `--yolo-place --trained-tabletop` in
[isaac/scripts/run_bottle_pick.py](../isaac/scripts/run_bottle_pick.py). Each
attempt obtains fresh domain-44 evidence; control uses a separate domain-46
simulation, not the frozen preview. The launcher checks the model hash, five
distinct observations, finite consistent targets and capture freshness. On
execution, the fingerprint must match current code/model/scene dependencies;
bottle XY change must be at most 2 mm, tray XY/floor combined change at most
2 mm, and rim change at most 2 mm. Legacy approvals do not authorize this mode.
See [robot_tabletop_command.md](robot_tabletop_command.md) for the full contract.

The unchanged superclass **baseline fixture guard runs first**. The real
simulated tray fixture must remain at its original world XY `[0.45, -0.25]`,
root Z `0`. Detected tray XY must then agree with the fixture within 8 mm and
the detected floor with the baseline 8 mm floor within 3 mm. Detecting another
tray position does not enable moving the fixture or bypassing that guard.

Control is not purely vision-based: simulator poses supply carry-offset
compensation, safety guards and final contact checks. Contact, slip, tilt,
tracking and support checks remain necessary. A passing perception test says
nothing about grasp stability, collision-free motion or full pickup/placement
success. **No such motion success was verified here.**

## Frozen preview and Play

The current [isaac/scripts/wrist_camera_ros.py](../isaac/scripts/wrist_camera_ros.py)
locks the camera preview to a **stopped timeline director** with automatic time
updates disabled. External Play requests are ignored before physics starts;
Play is not the way to enable robot motion in this camera-only preview. This
supersedes the older behavior where pressing Play could terminate the preview.

Rendering uses zero simulation-time steps. Calibration and camera TF are read
again for each capture. If the frozen hand pose or simulation time changes,
the live preview stops publishing potentially invalid images and keeps the
window open for inspection. The headless `--validate --test-play` regression
passed: Play was ignored, five camera frames were valid, simulation time stayed
zero and source scene layers were unchanged. Live ROS delivery was also verified.

## Reproducible CPU-only tests

Use the existing CPU environment; do not install or change packages. From the
repository root, the model-free suite is:

```bash
env -u PYTHONPATH -u TABLETOP_TEST_NPZ PYTHONDONTWRITEBYTECODE=1 \
  ./env_yolov9_cpu/bin/python -B -m unittest discover \
  -s tests -p test_tabletop_perception.py -v
```

Coverage includes exact class/label gating, ambiguity, ROI dimensions and
clipping, calibrated deprojection, valid and rejected synthetic trays, nonblue
obstacles, invalid model input, and RGB row padding/bad encodings. Messages are
`SimpleNamespace` objects; guarded script import rejects attempts to import
ROS, Isaac or Torch at import time. No simulator or model is needed for these
unit tests. Existing parser/approval tests remain separate in
[tests/test_robot_tabletop_command.py](../tests/test_robot_tabletop_command.py).

Optional recorded-frame regression, using only an explicitly trusted local NPZ:

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  TABLETOP_TEST_NPZ="$PWD/ros2_ws/log/bottle_rgbd_input.npz" \
  ./env_yolov9_cpu/bin/python -B tests/test_tabletop_perception.py RecordedModelTests -v
```

The variable must name an absolute existing local path. NPZ fields are `rgb`,
`depth`, `k` and `camera_to_world`; loading uses `allow_pickle=False`. The test
loads the pinned local model, checks CPU parameters and class mapping, detects
both targets and runs their actual RGB-D localization. Without the variable it
is skipped; once opted in, missing artifacts or detection failures fail the
test instead of silently skipping. It emits no image or detection-report files.
A recorded frame may be stale and is **never live capture or motion approval**.

### Results on 2026-10-21

- Model-free run after repair: **26 passed, one optional model test skipped**
  (27 methods collected). Parser/approval suite: **18 passed**.
- Fixed regression: `roi_mask((4, 7), [-5, 0, -2, 3])` and
  `roi_mask((4, 7), [0, -5, 6, -2])` return nonempty masks even though the boxes
  lie wholly outside the image. Negative upper bounds previously became Python
  from-the-end slice indices. Upper bounds are now clamped to zero and both
  regression assertions pass.
- Optional recorded model/RGB-D regression: **1 passed**, approximately 3.2 s,
  using the existing local bottle RGB-D recording and pinned CPU checkpoint.
- Nonfatal warnings: invalid multiplication while testing infinite depth, and
  an upstream `pkg_resources` deprecation during model import. No dependency
  changes were made to suppress them.

This is one offline regression frame, not an independent accuracy evaluation,
live ROS synchronization test, Play-lock test, or complete motion validation.