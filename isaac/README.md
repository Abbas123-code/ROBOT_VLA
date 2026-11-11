# isaac/ — The Body

Everything related to **NVIDIA Isaac Sim** (the simulated world + robot).

- `scenes/`  — USD stages / scene descriptions (the virtual room, table, objects).
- `scripts/` — standalone Python scripts that drive Isaac Sim.
- `assets/`  — robot models and object meshes (large binaries; git-ignored).

## Available workflows

- [Panda red-cube pick and place](../docs/panda_pick_place.md): validation-gated simulation.
- [Cluttered tabletop](../docs/cluttered_tabletop.md): multi-object scene construction.
- [Virtual wrist RGB-D camera](../docs/wrist_camera.md): frozen sensor preview.
- [Trained detection and moving video](../docs/trained_wrist_detection.md): camera-guided bottle validation and separate RGB diagnostics.
- [Dataset capture through CVAT and training](../docs/dataset_to_robot.md): full data workflow.

Generated scenes and downloaded assets are local, Git-ignored artifacts. Restore
them before launching on a prepared Isaac Sim host. Physics validation can move
the simulated robot; do not run competing controllers or reuse another machine's
validation as execution approval.
