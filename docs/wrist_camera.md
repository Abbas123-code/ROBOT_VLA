# Panda wrist RGB-D camera

The [publisher](../isaac/scripts/wrist_camera_ros.py) adds a virtual camera under
the Panda `panda_hand` prim in the **session layer** of the cluttered scene.
No saved scene is overwritten. The existing remote Orbbec asset is preserved;
this publisher uses its own ideal pinhole camera, not that asset's sensor model.

## Open the camera and RViz

In VS Code, select **Terminal → Run Task**:

1. **Panda: wrist RGB-D camera (no motion)**
2. **Panda: RViz wrist camera**

In RViz's Displays list, tick **Wrist RGB** and **Wrist depth (metres)** if they
are unchecked. The installed RViz can disable these image docks at startup
despite the saved enabled flags. Press **Ctrl+S** after arranging the panels to
save your preferred layout. The central 3D view shows coordinate frames; the
camera pictures appear in the image panels, not as a robot model in that view.

The RViz configuration is [configs/wrist_camera.rviz](../configs/wrist_camera.rviz).
Close the dedicated camera window to stop its publisher. Existing simulator
windows and pick-place controllers are not modified or restarted.

## What is published

ROS domain **44**, middleware `rmw_fastrtps_cpp`, node `wrist_camera_publisher`:

| Topic | Message / meaning |
|---|---|
| `/wrist_camera/color/image_raw` | `sensor_msgs/Image`, `rgb8`, 640 × 480 |
| `/wrist_camera/depth/image_raw` | `sensor_msgs/Image`, `32FC1`, optical-axis depth in metres |
| `/wrist_camera/color/camera_info` | Pinhole calibration |
| `/wrist_camera/depth/camera_info` | Same calibration; registered depth |
| `/tf` | `world → panda_hand → wrist_camera_optical_frame` |

Both images and calibration use the same capture timestamp. Invalid depth is
NaN. Optical axes are +X right, +Y down, +Z forward. Focal lengths are
`fx = fy = 320` pixels; principal point is `(320, 240)`. No lens distortion is
modelled. Captures are throttled to at most approximately 5 Hz; actual rate
depends on rendering. This frozen-scene preview uses **wall-clock timestamps**,
not `/clock`; keep RViz `use_sim_time` false.

The eye is at local hand offset `(0.08, 0, 0.035)` metres, aimed toward the table
once at startup. Its resulting transform is then fixed relative to the hand.
There is no physical camera housing, mass, or collision shape. USD parenting
makes it a wrist-mounted sensor, but following a moving physics articulation
has **not** been tested by this preview.

The viewport can edit the sensor's pose. The publisher now reads its current
pose and intrinsics each capture, rather than continuing to publish startup
values after viewport navigation. Keep the camera still when collecting a
manipulation snapshot. See [bottle detection](bottle_detection.md) for the
working RGB-D subscriber and the aborted, unvalidated bottle-grasp experiment.

## Safety and limitations

- Physics time stays at zero. No Play, joint commands, grasping or pick-place.
- Do not press Play in this preview; the script stops on unexpected playback.
- This publisher performs no recognition itself. A separate model-specific
	RGB-D bottle detector now works; bottle grasping is **not validated**.
- This does not validate paths around clutter or a physical mounting bracket.
- No driver, system CUDA or GPU setting changes were made.

## Validation

Five rendered RGB-D frames passed the headless capture check; all table objects
and the receiving box were visible in the saved RGB image. Local source-file
hashes remained unchanged. Remote asset URIs are not local files and are excluded
from the on-disk hash check.

The independent [ROS subscriber test](../tests/check_wrist_camera_ros.py) received
three fresh RGB and depth frames, calibration and both TF edges, checked
encodings, dimensions, synchronized timestamps and valid depths. Approximately
90–91% of pixels had finite in-range depth in the tested view.

- [Capture report](../ros2_ws/log/wrist_camera_validation.json)
- [Live preview report](../ros2_ws/log/wrist_camera_preview.json)
- [ROS test report](../ros2_ws/log/wrist_camera_ros_validation.json)
- [Captured RGB image](../ros2_ws/log/wrist_camera_rgb.png)

The user reported both RViz images appearing after manually enabling the image
displays. Automated RViz checks still saw disabled image displays; startup
auto-enabling is therefore **not verified**. TF visualization was observed.