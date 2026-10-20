# Portable setup: home PC

## Scope and safety

The Docker image is **CPU-only offline vision tooling**, not an Isaac Sim or CVAT
replacement. It does not move the robot or train a model. No host GPU drivers,
CUDA, existing environments or CVAT services are modified. Supported/tested target:
Linux x86_64 with Docker Engine, Docker Compose v2+ and GNU Make.

Do not copy Python virtual environments between PCs. Install Isaac separately on
a compatible system when you need simulation; the CPU tools need no NVIDIA GPU.
Windows/macOS and ARM hosts have not been tested. WSL2 Linux may work but is not
certified here. Initial Docker builds require internet and several GB of space.

## 1. Choose how to transfer

### Complete project copy (recommended for your home PC)

From the repository root, `make bundle` creates a timestamped archive and SHA256
sidecar under `transfer/`. If this system has no `python3`, supply the correct
Python 3.11+ executable with Make's `PYTHON` variable. This uses only stdlib.

The archive contains code/config/docs, table images and any labels saved beneath
`tableimage/`, data/checkpoints/datasets, local scene assets, and ROS source.
It excludes environments, Git history, caches and old motion validation reports.
Files are not deleted or moved. Existing scene files are never rewritten.
Local absolute USD references are made relative **inside the archive only**.
The archive contains `TRANSFER_MANIFEST.json` with per-file hashes and rewrites.

Copy the archive and its checksum file to your home PC. Verify the archive using
`sha256sum -c` with the sidecar filename, then extract it using your file manager
into a **new empty directory**. Do not extract over existing work or annotations.

**CVAT annotations still stored inside CVAT are NOT in this archive.** Export
your CVAT task/dataset separately before migration, or follow CVAT's official
backup/restore procedure for its database and volumes. Never delete its volumes.

### Git for code history

Git is for small project files; it is not the complete dataset/scene backup.
The public repository is [Abbas123-code/ROBOT_VLA](https://github.com/Abbas123-code/ROBOT_VLA).
Keep private data and credentials out of all commits.

1. Review `git status --short` and `git diff`.
2. Use `git add --dry-run .` to inspect what would be staged. Review for secrets.
3. Stage the intended files, review `git diff --cached`, and commit.
4. Verify the configured remote and push only when ready.
5. On the other PC, clone the repo and separately copy data/assets from the bundle.

Ignoring a file does not untrack files already committed. The generated LaTeX
SyncTeX file is no longer tracked, but its local copy and historical commit are
preserved. Documentation PDFs remain in Git.

## 2. Build and test CPU Docker

From the extracted/cloned repository root:

| Command | Purpose |
|---|---|
| `make build` | Download dependencies, pinned official source and checksum-verified weights |
| `make doctor` | Verify CPU torch/torchvision, clean pinned YOLO source and checkpoint |
| `make inspect` | Check images; report duplicate pixels and quality exclusions |
| `make detect` | Run inference on the default close-up bottle image |
| `make detect IMAGE=/input/table_000.png OUTPUT=/output/run-002` | Choose another input and fresh output folder |

`make detect` deliberately refuses to overwrite an existing output folder. Choose
a new `OUTPUT` for each run. Results appear in `outputs/<run>/detections.png` and
`detections.json`. These are **predictions, not manual labels**, and can be wrong.

Make passes your Linux UID/GID so output files belong to you. If invoking Compose
directly, set `LOCAL_UID`/`LOCAL_GID` to your user/group IDs (defaults are 1000).
The Compose service mounts `tableimage/` read-only and `outputs/` writable. It
does not bind the entire project, does not mount a Docker socket, has no network,
uses a read-only root filesystem, and exposes no ports. Temporary run containers
are removed by `--rm`; the image remains reusable.

CPU model versions: PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, official YOLOv9-S.
The source revision and checkpoint hash are in `configs/yolov9_official.json`.
The image preserves the upstream checkout/license. It uses no Ultralytics package.
Top-level inference requirements are pinned; the base image tag and transitive
packages are not fully locked, so builds are not guaranteed bit-identical.

## 3. Isaac Sim and ROS on the destination

The present project was developed on Ubuntu 24.04, ROS 2 Jazzy and Isaac Sim
6.2.0.0. Check NVIDIA's hardware, OS, driver and license requirements before
installing Isaac on your home PC. **Do not run this CPU image as a simulator.**

- Recreate the host Isaac environment; do not reuse this PC's venv paths.
- Restore `isaac/scenes/` and `isaac/assets/` from the portable bundle.
- The user-added Orbbec Gemini2 USD still references an NVIDIA HTTPS asset and
  requires network access (not silently downloaded/relicensed by the exporter).
- Install/source ROS Jazzy and build `ros2_ws/src` in the destination workspace.
- The motion launchers still expect `env_isaacsim` and `env_yolov9_cpu` beneath
  the project root. Docker's offline CLI does not replace these live-control envs.
- Revalidate every robot task in an isolated simulation. Do not transfer an old
  success report as authorization to execute on another PC. Bottle placement
  remains unvalidated even on the original PC.
- The frozen wrist camera publishes without pressing Play; Play invalidates its
  safety checks and can close that preview.

## 4. Folder organization

| Location | Purpose / migration policy |
|---|---|
| `src`, `scripts`, `isaac/scripts`, `ros2_ws/src` | First-party code; Git |
| `configs`, `docs`, root deployment files | Config and instructions; Git |
| `tableimage` | Original images, labels and dataset guides; data in bundle |
| `tableimage/zoom_views` | Extra near/medium/far images; read quality exclusions |
| `outputs` | Disposable prediction outputs; not Git or migration approval |
| `transfer` | Timestamped migration archives; not Git |
| `isaac/assets`, `isaac/scenes`, `data` | Large assets/data; migration bundle |
| `env_isaacsim`, `env_yolov9_cpu` | Host-only environments; recreate, never Git |
| `third_party/yolov9` | Re-fetched pinned source; Docker handles this |
| `IMAGES`, `old realtime.py`, pasted screenshot | Preserved local legacy files; not packaged |

The cleanup is intentionally non-destructive. No `git clean`, environment
deletion, dataset renaming, CVAT shutdown or volume removal is performed.

## Verification on the original PC (2026-10-20)

- Docker image built successfully; build context was only 31.68 kB rather than
  sending the 27 GB Isaac environment or dataset into the build.
- `make doctor`: CPU torch/torchvision and official source/checkpoint verified;
  no Ultralytics package.
- `make inspect`: image inventory ran successfully. The dataset has changed since
  initial capture: 148 image files were present, with duplicate copies reported.
  Nothing was deduplicated or deleted because annotations may refer to filenames.
- `make detect`: offline inference wrote image and JSON outputs; a second run
  correctly refused to overwrite the existing output directory.
- Container `pip check`: no broken requirements.
- Three stdlib exporter regression tests passed (`tests/test_portable_bundle.py`).
- Migration archive checks: per-file hashes verified, local USD references resolve
  within the archive, and the current source scene files remain unchanged.
- Existing CVAT UI remained running. No home-PC or new Isaac installation has
  been tested; full simulation compatibility still needs destination validation.