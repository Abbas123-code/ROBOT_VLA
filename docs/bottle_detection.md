# Bottle detection — motion stopped by user request

**Later update:** The user reauthorized work using only the official YOLOv9
repository. [Official YOLOv9 results](official_yolov9.md) describe working live
recognition and three additional isolated grasp trials. Placement still failed
the slip guard; no visible execution occurred. The original colour-only trial
below is historical.

## Working: subscribe to RGB and locate the bottle

[The detector](../scripts/detect_bottle_rgbd.py) subscribes to the wrist camera on
ROS domain **44**. It matches RGB/depth timestamps, uses camera calibration and
the exact-time optical-frame TF, and requires five consistent bottle estimates.
It sends **no robot commands**.

Detection uses a cyan/blue colour component and depth, then fits the known
64 mm upright cylindrical bottle body. It rejects the low blue box using 3D
height, rejects ambiguous candidates, and requires a sufficient visible arc
and small fitting residuals. This is a model-specific colour/geometry detector,
not general AI bottle recognition. It assumes an upright opaque bottle on the
world Z=0 table. Production detection does not read the bottle's USD pose.

With the wrist-camera task running, source ROS Jazzy and run the detector:

```bash
source /opt/ros/jazzy/setup.bash
./env_isaacsim/bin/python scripts/detect_bottle_rgbd.py
```

Verified image estimate: approximately **X=0.72245 m, Y=0.34204 m** in world
coordinates. Five-frame spread was about 0.01 mm in the stationary rendered
scene; this is repeatability, **not absolute accuracy**. The separate simulation
safety check measured approximately **3.2 mm** error against the actual bottle.

- [Detection report](../ros2_ws/log/bottle_detection.json)
- [Annotated RGB image](../ros2_ws/log/bottle_detection.png)
- [Detector implementation](../src/perception/bottle_rgbd.py)

### Camera-data correction

The first detection attempt rejected inconsistent heights. The camera preview
allows viewport navigation, but its publisher previously kept the initial TF
and intrinsics. The publisher now reads the current camera transform and
intrinsics for each capture. After restarting only that camera publisher,
five-frame localization passed. Do not move the camera during a capture intended
for manipulation; detections are snapshots, not visual servoing.

## Not validated: bottle pickup

A separate domain-45 headless test attempted a slow approach, body grasp and
5 cm lift using the RGB-D target. Simulator object poses were used only for
safety checks and outcome verification, not to correct the target.

1. A 23 cm top-down hover was rejected by IK at the far target.
2. A reachable 22 cm hover needed shorter approach segments to pass the existing
   joint-continuity limit. Neither limit was weakened.
3. The segmented approach reached the bottle, but during descent the **palm
   collided with the bottle cap**. The contact guard aborted before finger
   closure or lift.

The user then selected **Stop after bottle detection**. No further grasp tests
and no visible bottle pickup were run. No bottle was teleported, attached to the
gripper or moved closer to make the attempt succeed.

- [Failed isolated test report](../ros2_ws/log/bottle_pick_validation.json)
- [Experimental controller](../isaac/scripts/bottle_pick_controller.py)
- [Experimental launcher](../isaac/scripts/run_bottle_pick.py)

**Do not treat these motion scripts as ready to execute.** The launcher requires
a successful matching validation report before visible execution; the current
report has `completed: false`. There is no automatic bottle-pick VS Code task.
A different grasp requires new explicit authorization and isolated validation.

Original cube pick-place code/scenes remain unchanged. No GPU settings, drivers
or system CUDA were changed.