# src/ — The Brain

Our own Python code. Each subfolder is introduced in a specific phase.

- `utils/`      — transform & math helpers (Phase 1).
- `control/`    — forward/inverse kinematics, controllers (Phases 2, 5).
- `perception/` — computer vision: detect the object, find its pose (Phases 6-7).
- `vla/`        — the Vision-Language-Action model (Phases 9-16).

We build the **classical** pipeline first, then gradually replace pieces with
learned models.
