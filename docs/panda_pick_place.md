# Automated red-cube → blue-box task

Added 20 October 2026. This supersedes the older guide's statement that no
automated placement sequence exists. Simulation only; not a real-robot controller.

## Run it

In VS Code: **Terminal → Run Task → Panda: validated automatic pick and place**.

This opens a **new dedicated Isaac window**, loads the saved box scene, initializes
the robot at HOME with open fingers, and runs the sequence once. It does not
control the already-open manual Isaac window or restore yesterday's held cube.
Do not start the old oscillation demo or another manipulation script alongside it.

The complete test must pass first. A fingerprint of the controller, launcher,
kinematics, URDF and composed scene layers is checked before visible execution.
If those files change, revalidate. Reports are overwritten on subsequent runs.

From the project root, in a Bash terminal:

```bash
source /opt/ros/jazzy/setup.bash
./env_isaacsim/bin/python isaac/scripts/run_panda_pick_place.py --validate
```

After `PICK_PLACE_VALIDATION_PASSED`, the visible run is:

```bash
source /opt/ros/jazzy/setup.bash
./env_isaacsim/bin/python isaac/scripts/run_panda_pick_place.py --execute
```

No GPU configuration, driver or system CUDA changes are required. The original
scene files are not saved/overwritten. A process lock prevents two task instances.
Close the dedicated task window before running the task again.

## One coordinated ROS node, not competing controllers

- Node: `/vla_pick_place/panda_pick_place`.
- ROS domain: **43**, separate from the historical demo's domain 42.
- Middleware: `rmw_fastrtps_cpp`.
- `/vla_pick_place/status`: `std_msgs/msg/String`, JSON phase events, retained
  latest event for late subscribers.
- `/vla_pick_place/joint_states`: `sensor_msgs/msg/JointState`, measured joints
  about 10 Hz, timestamps are simulation time. No global `/clock` is published.
- `/vla_pick_place/pause`: `std_srvs/srv/Trigger`, abort-and-pause request. There
  is intentionally no automatic resume/retry/release service.

The node runs inside the Isaac Python process so it can read **actual** cube
poses, finger contacts and measured joint states. It directly controls USD drives.
It does not publish into the old `/vla_sim/joint_command` topic. The box scene's
legacy OmniGraph remains disabled. A separate demo commander is neither required
nor appropriate; planning, grasp, transport, release and monitoring have one owner.

Read-only monitoring from another terminal:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=43
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 node list
ros2 topic echo /vla_pick_place/status std_msgs/msg/String --qos-durability transient_local
```

To request a stop during the task (same terminal environment):

```bash
ros2 service call /vla_pick_place/pause std_srvs/srv/Trigger '{}'
```

Closing/pausing the dedicated Isaac window also interrupts the task. On failure,
physics is paused and the scene remains visible. **Do not press Play to retry:**
the script has aborted and its last motor targets can still exist. Diagnose the
failure, close the dedicated task window, and start a new validated run instead.

## Sequence and verified geometry

1. Settle fresh baseline: base near world zero, cube centre near `[0.5, 0, 0.025]` m.
2. Hover over cube, grasp frame target Z = 0.15 m.
3. Descend with open fingers, target Z = 0.030 m.
4. Close slowly, 5 N drive limit per finger; require sustained bilateral contact.
5. Lift to grasp target Z = 0.18 m.
6. Move horizontally above the box, centred at X/Y = `[0.45, -0.25]` m.
7. Lower inside the opening to grasp target Z = 0.040 m.
8. Open gently, allowing the last few millimetres to settle onto the floor.
9. Verify the released cube rests inside the box with bottom contact.
10. Retreat upward with open fingers, hold, and verify the cube did not follow.
11. Pause physics on success so the result is easy to inspect.

Box outer dimensions: 20 × 20 × 6 cm. Opening: 18.4 × 18.4 cm. Five separate
static panels, no lid or filled interior. Floor top Z = 0.008 m; resting 5 cm
cube centre Z = 0.033 m. The robot is not attached to the cube by any constraint.

## Validation evidence

The first complete isolated test passed all phases in about 91.25 seconds of
simulation time (including initial settling):

- Cube final position: `[0.45019016, -0.24893329, 0.03299996]` m.
- Box-floor contact confirmed; fingers open; hand retreated.
- Peak joint tracking error: 0.00967 rad.
- Peak relative slip while carrying: 0.000117 m (0.117 mm).

The subsequent **visible execution also completed successfully** on 20 October:
its separate report records `completed: true`, cube inside the box, bottom contact,
open fingers and a completed retreat, with the same final pose and measured
tracking/slip values above. Physics was paused afterward. The ROS node and its
status publisher were independently discovered on domain 43 after completion.
The one-shot filtered ROS CLI status read did not return output; the visible
outcome was verified from the execution report, not that CLI subscription.

- [Validation report](../ros2_ws/log/panda_pick_place_validation.json).
- [Visible execution report](../ros2_ws/log/panda_pick_place_execution.json).
- [Controller](../isaac/scripts/pick_place_controller.py).
- [Launcher](../isaac/scripts/run_panda_pick_place.py).
- [Box scene](../isaac/scenes/panda_tabletop_with_box.usda).
- [Box construction](../isaac/scripts/placement_box.py).
- [Non-destructive box scene builder](../isaac/scripts/build_panda_box_scene.py).
- [Isolated box-floor test](../tests/check_panda_box.py).

Guards cover clock/stage changes, pose mismatch, finite values, joint limits,
tracking, IK discontinuities, cube tilt/slip, unexpected reported contacts,
transfer clearance, release pose and final supported placement. Self-collision
checking is enabled; only fixed-joint-connected overlapping imported parts are
filtered. This is finite-step validation of these assets, not a general collision
proof, general-purpose planner or hardware safety system. Changing the box,
cube, scene, load, friction or start configuration requires new validation.

## Historical update

The terminal log supplied on 20 October confirmed yesterday's additional 5 cm
live lift succeeded (cube rise 0.04980946 m; final centre Z = 0.09469093 m).
That process subsequently exited with code 137. Its live state was lost; the
exit code alone does not establish why it was killed. The new task does not
depend on that state. The log also records closure displacement 0.00113899 m
(1.139 mm), correcting the older screenshot transcription of 0.114 mm.