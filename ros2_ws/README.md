# ros2_ws/ — The Nervous System

A standard **ROS 2 Jazzy** workspace. It carries messages between programs
(camera → perception → control → robot).

- `src/` — our ROS 2 packages live here.

Typical build (later, from this folder):

```bash
colcon build
source install/setup.bash
```

`build/`, `install/`, and `log/` are generated and git-ignored.
Used from Phase 3 onward.
