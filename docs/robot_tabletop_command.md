# Text command: trained tabletop bottle placement

This is a deterministic **allowlist**, not an LLM planner or a general robot
language interface. It controls only a separate Isaac simulation (domain 46),
not a physical robot or the existing frozen camera scene (domain 44).
No simulation was run to validate this integration. Unit tests cover the parser
and launcher boundary; they do not establish collision freedom or grasp success.

## Accepted commands

Canonical text: `pick up the bottle and place it in the blue tray`.

The parser accepts case/whitespace changes, optional leading `please`, optional
`the`, `put` instead of `place`, `into` instead of `in`, a final `.` or `!`, and
the common typo `blue track` instead of `blue tray`. It matches the **entire**
command. Other objects, destinations, negation, appended tasks and shell syntax
are rejected. There is no microphone/listening mode.

Exactly one explicit mode is required. `--validate` does not execute afterward;
`--execute` never automatically validates or retries. The text is never passed
to a shell or evaluated. A constant Bash program sources ROS Jazzy and executes
the launcher using quoted positional arguments (`exec "$@"`).

## Tests without ROS or simulation

From the repository root:

```bash
./env_isaacsim/bin/python -m unittest discover -s tests -p test_robot_tabletop_command.py -v
./env_isaacsim/bin/python scripts/robot_tabletop_command.py --help
./env_isaacsim/bin/python isaac/scripts/run_bottle_pick.py --help
```

These tests mock all child-process launches. They do not start a camera,
capture process, ROS node, or simulator.

## Opt-in simulation commands (not run during implementation)

Requires ROS 2 Jazzy, both existing Python environments, the baseline cluttered
scene and its assets, and a fresh domain-44 wrist camera publisher. Close any
previous dedicated bottle-control window before another attempt; the shared
bottle launcher lock rejects concurrent sessions.

```bash
# Starts isolated HEADLESS simulation; not a syntax-only check:
./env_isaacsim/bin/python scripts/robot_tabletop_command.py --command 'pick up the bottle and place it in the blue tray' --validate

# Only after successful validation and explicit authorization for visible simulation:
./env_isaacsim/bin/python scripts/robot_tabletop_command.py --command 'pick up the bottle and place it in the blue track' --execute
```

The CLI calls the Isaac interpreter with
`run_bottle_pick.py --yolo-place --trained-tabletop --validate` (or `--execute`).
The new flag requires `--yolo-place`. Omitting it preserves the original COCO
detector/capture route and its report namespace.

## Capture contract and execution approval

The launcher runs the CPU interpreter with `scripts/wrist_yolo_ros.py --capture`
and reads `ros2_ws/log/tabletop_detection.json`. Missing dependencies or invalid
evidence stop the run; there is **no COCO, color, or fixed-coordinate fallback**.
Capture has a 90-second subprocess timeout. A completed report must come from
the new capture attempt, must not be future-dated, and must be at most 15 seconds
old when read (120 seconds allowed after Isaac startup).

Required top-level fields:

- `completed: true`, finite `captured_at_unix`, finite two-element bottle `center_xy`.
- Finite `tray_center_xy`, `tray_floor_z`, and `tray_rim_z > tray_floor_z` (world metres).
- `detector: "trained-yolov9-tabletop"`.
- `weights_sha256: "28db4ee45eb19eb6d44d7cb9bcc8d0c13b8a4fda95f57dcd89dd508ef9910841"`.
- Exactly five `frames`. Each contains `detector: "WongKinYiu/yolov9"`, distinct
  `image_stamp: [sec, nanosec]`, bottle `center_xy`, `tray_center_xy`, `tray_floor_z`,
  `yolo_bottle: {label: "Water Bottle", class_id: 6, confidence: >=0.5}`, and
  `yolo_tray: {label: "Blue Tray", class_id: 0, confidence: >=0.5}`. Confidence
  must be finite and at most 1. Frame targets must agree with report targets
  within 3 mm. Exact publisher RGB/depth/calibration/TF timestamp matching is
  owned by the capture/camera integration, not this launcher.

The local training artifact `outputs/training/tabletop_yolov9s_v1/weights/best.pt`
must match the pinned hash. The trained fingerprint includes that hash, the
existing motion/scene dependencies, `configs/yolov9_tabletop.json`,
`src/perception/yolov9_tabletop.py`, `src/perception/tabletop_rgbd.py`,
`scripts/wrist_yolo_ros.py`, `isaac/scripts/wrist_camera_ros.py`, and the text CLI.
Those perception/config/camera files are owned by the separate integration;
this change does not create or modify them. Missing files fail closed.

New reports are `ros2_ws/log/bottle_tabletop_place_validation.json` and
`ros2_ws/log/bottle_tabletop_place_execution.json`. Existing `bottle_yolo_place`
approvals cannot authorize this mode. Execution requires successful validation,
an identical fingerprint/model, bottle XY change at most 2 mm, tray XY/floor
combined change at most 2 mm, and rim change at most 2 mm. Every attempt gets
a fresh capture. Beginning a new validation invalidates any older successful
trained validation, even if the new capture fails.

## Motion scope and limitations

- The **unchanged superclass baseline guard runs first**. It still requires the
  physical fixture at its original world pose (XY `[0.45, -0.25]`, root Z `0`).
  This feature does **not** support moving the actual tray to a new location.
- After that guard, detected tray XY must be within 8 mm of the actual fixture
  XY; detected floor must be within 3 mm of the known baseline floor `0.008 m`.
  Rejection never substitutes simulator/fixed target coordinates.
- Transfer/lowering/placement checks use detected tray XY, floor and rim. Rim
  clearance can only strengthen the previous minimum. The release gap remains
  4 mm. Contact, 4 mm slip, tilt, tracking, neighbour-motion, support and pause
  guards remain unchanged, as do finger forces and materials.
- This is **not perception-only control**: simulator bottle and grasp poses
  measure an XYZ carry offset after closure. That offset actively compensates
  the lowering target. Simulator state also supplies guards and final contact
  verification. Reports explicitly include this scope, detected tray geometry,
  model hash and measured carry offset. There is no visual servoing.
- Floor detection error can still fail the unchanged 2 mm final support check;
  passing the 3 mm preflight guard does not guarantee successful placement.
- Exceptions pause the control simulation. No automatic retry, release or
  restart is added. Validation must succeed on the current integration before
  visible execution can be approved. No grasp or motion success is claimed by
  parser/unit-test results.