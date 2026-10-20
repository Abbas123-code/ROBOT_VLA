# Tabletop objects — 20 October 2026

Open [the cluttered scene](../isaac/scenes/panda_tabletop_cluttered.usda) in Isaac,
or use **Terminal → Run Task → Panda: preview cluttered tabletop (no motion)**.
The preview opens in a separate window with physics stopped, the old ROS graph
disabled, and no robot controller. It does not change an existing GUI session.

The scene preserves the original Panda, red cube and blue receiving box and adds:

| Object | Size | World X/Y (m) | Mass |
|---|---|---|---|
| Green cube | 5 cm side | 0.70, 0.14 | 50 g |
| Black cube | 5 cm side | 0.73, −0.07 | 50 g |
| Water bottle | About 6.5 cm wide, 20.2 cm tall | 0.72, 0.34 | 250 g |
| Orange can | 5.8 cm diameter, 11.2 cm tall | 0.93, 0.18 | 120 g |
| Yellow block | 9 × 4.5 × 4 cm | 0.92, −0.16 | 80 g |

All five are dynamic rigid bodies with collision geometry. The bottle is an
original procedural model with a light-blue body, blue label and white cap—not
a liquid simulation, transparent water rendering, or separately removable cap.
No external assets or packages were downloaded. The red cube and blue box are
still the original task objects; the new objects are not yet robot targets.

## Validation

- Static bounds checked against table edges, other props, saved Panda collision
  meshes, the red cube and blue box.
- Five props stayed upright and on the table during a four-second isolated
  physics test. Robot and ROS graph were removed **before** test playback.
- A nonuniformly scaled sphere warning was resolved by using a real spherical
  bottle shoulder with matching visual/collision geometry, then retested.
- [Validation report](../ros2_ws/log/tabletop_props_validation.json).

**No robot movement or pick-and-place trajectory was tested in this new scene.**
Static clearance is not path clearance. The existing automatic pick-place task
still uses its original box scene, not this cluttered scene. A future run among
these objects needs new trajectory/contact validation; do not just switch scenes
under a running controller.

## Files and reproduction

- [Prop geometry and layout](../isaac/scripts/tabletop_props.py).
- [Scene builder and preview](../isaac/scripts/build_cluttered_tabletop.py).
- [Physics test](../tests/check_tabletop_props.py).

From the project root, build only if the output scene does not already exist:

```bash
./env_isaacsim/bin/python isaac/scripts/build_cluttered_tabletop.py
```

The builder refuses to overwrite an existing output. It layers on the box scene
and does not rewrite either baseline scene. The layered references use absolute
paths; keep the baseline scenes/assets and update references if moving projects.
Generated USD scenes are ignored by Git, so preserve the source generators too.

Run the isolated test:

```bash
./env_isaacsim/bin/python tests/check_tabletop_props.py
```

Open the stopped preview:

```bash
./env_isaacsim/bin/python isaac/scripts/build_cluttered_tabletop.py --preview
```

No driver, GPU settings, or system CUDA changes were made.