# ROBOT_VLA

Building a **Vision-Language-Action (VLA)** robot from first principles, in
**NVIDIA Isaac Sim + ROS 2 Jazzy**.

> Mental model of the whole project: **Body (Isaac Sim), Nerves (ROS 2), Brain (our code → VLA).**
> See [docs/00_the_basics.md](docs/00_the_basics.md) for the plain-language foundations.

---

## Folder map

| Folder | Role in the mental model | What lives here |
|--------|--------------------------|-----------------|
| `docs/` | — | Phase-by-phase learning notes. Start with `00_the_basics.md`, then `phase0_environment.md`. |
| `isaac/` | **Body** | Isaac Sim scenes (`scenes/`), standalone Python scripts (`scripts/`), robot/object assets (`assets/`). |
| `ros2_ws/` | **Nerves** | ROS 2 Jazzy workspace. Our packages go under `ros2_ws/src/`. |
| `src/` | **Brain** | Our Python code (below). |
| `src/utils/` | Brain | Transform & math helpers (Phase 1). |
| `src/control/` | Brain | FK / IK / controllers (Phases 2, 5). |
| `src/perception/` | Brain | Computer vision (Phases 6-7). |
| `src/vla/` | Brain | The VLA model itself (Phases 9-16). |
| `data/datasets/` | — | Recorded demonstrations (Phases 12-13). |
| `data/checkpoints/` | — | Trained model weights. |
| `configs/` | — | YAML/JSON configuration files. |
| `notebooks/` | — | Math practice & experiments. |
| `scripts/` | — | Helper shell/Python scripts. |
| `tests/` | — | Automated tests. |

## Roadmap

Phase 0 (environment verification) → Phase 18 (advanced topics).
Progress is tracked as we go. We do **not** jump ahead: understand each phase,
implement it, verify the output, then move on.

## Current status

- **Phase 0 — Environment verification** (in progress)

See [docs/phase0_environment.md](docs/phase0_environment.md).
