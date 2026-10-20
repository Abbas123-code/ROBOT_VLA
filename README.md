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

- Isaac Sim Panda/table scene, ROS camera streams and cube pick/place experiments exist.
- Official **WongKinYiu/yolov9** CPU inference is implemented; no Ultralytics package.
- Bottle perception is unreliable on some synthetic views. Bottle placement is **not validated**.
- 136 unlabelled table images are available locally, including close-ups. Three new
	views are flagged for exclusion. Manual labels and model training are still pending.

## Use on another PC

Start with **[the portable setup guide](docs/portable_setup.md)**.

| Component | How it runs | Included here |
|---|---|---|
| Offline YOLO inference / image inspection | CPU-only Docker | Dockerfile, Compose, Make targets |
| Dataset and saved scenes/assets | Separate transfer archive | Non-destructive export script |
| Isaac Sim + live ROS robot simulation | Compatible host installation | Scripts; **not** in CPU Docker |
| CVAT | Existing independent installation | No containers or database changed |

Convenience targets: `make build`, `make doctor`, `make inspect`, `make detect`,
`make bundle`. The Docker runtime mounts images read-only and writes predictions
under `outputs/`. It has no network access, GPU devices, ROS control or exposed ports.
Internet is required for the initial image build.

## Quick start: clone and run

### 1. Requirements

For the **CPU Docker tools**, use a Linux x86_64 PC with:

- Git, GNU Make, Docker Engine and Docker Compose v2 or newer.
- A running Docker daemon that your user can access.
- Internet for the first build and several GB of free disk space.

No NVIDIA GPU, Isaac Sim, ROS or local Python environment is needed for this
Docker workflow. Install Docker using the [official Linux instructions](https://docs.docker.com/engine/install/)
if it is missing. Check your installation:

```bash
git --version
make --version
docker version
docker compose version
```

### 2. Get the project

**Option A — Clone the public Git repository.**
The project repository is [Abbas123-code/ROBOT_VLA](https://github.com/Abbas123-code/ROBOT_VLA).

```bash
git clone https://github.com/Abbas123-code/ROBOT_VLA.git ROBOT_VLA
cd ROBOT_VLA
```

Git does **not** include images, model checkpoints, local scene assets or virtual
environments. To use the existing dataset, also obtain the transfer bundle.
Extract it into a separate empty folder, then copy its `tableimage/` contents
into the checkout, keeping existing images and annotations. For simulation,
also restore `isaac/scenes/`, `isaac/assets/` and the needed `data/` contents.
Do not overwrite newer source files in your clone with older archived code.

**Option B — Use the complete transfer bundle without Git.** Copy
`ROBOT_VLA-home-pc.tar.gz` and its `.sha256` sidecar to your home PC. In the folder
containing both files, verify and extract into a new temporary directory:

```bash
sha256sum -c ROBOT_VLA-home-pc.tar.gz.sha256
DEST=$(mktemp -d "$HOME/robot-vla-import.XXXXXX")
tar -xzf ROBOT_VLA-home-pc.tar.gz -C "$DEST"
cd "$DEST/ROBOT_VLA"
```

Continue only if checksum verification reports `OK`. The bundle includes code,
local images/assets and portable scene references, but not Git history or Python
environments. It is a snapshot: generate a new bundle on the original PC with
`make bundle` after further code, image or label changes. The previously created
home-PC archive predates this README expansion; a Git checkout or new export
contains the latest guide.

### 3. Build and check the CPU container

Run these commands from the project root—the folder containing this README:

```bash
make build
make doctor
make inspect
```

- `build` installs CPU dependencies and downloads the pinned official
	WongKinYiu YOLOv9-S source and weights. No Ultralytics package is used.
- `doctor` should report `"completed": true` and `"device": "cpu"`.
- `inspect` reads the dataset and reports duplicate images and flagged exclusions.
	It does not delete, label or change images. It needs images restored in step 2.

### 4. Run object detection on an image

After restoring the dataset, run the default bottle close-up:

```bash
make detect
```

To choose another image and output directory:

```bash
make detect IMAGE=/input/table_000.png OUTPUT=/output/run-002
```

Container paths map to your PC as follows:

| Container path | Local project folder |
|---|---|
| `/input/` | `tableimage/` (read-only) |
| `/output/` | `outputs/` (writable) |

Open `outputs/bottle-near/detections.png` for the default run, or
`outputs/run-002/detections.png` for the second example. The matching
`detections.json` contains classes, scores, bounding boxes and model settings.
**Predictions can be wrong; they are not training labels or motion commands.**

Choose a new output folder on each run: overwriting existing results is refused.
Each command exits and removes its temporary container automatically. You do not
need `docker compose up`, a web port, or a background service for these tools.

## Optional: run Isaac Sim and the live wrist camera

This is a **separate host workflow**, not something the CPU container can run.
The original development setup used Ubuntu 24.04, ROS 2 Jazzy, Python 3.12 and
Isaac Sim 6.2.0.0. The home PC must meet NVIDIA's current compatibility and license
requirements. Follow the [Isaac Sim installation documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/index.html)
and [ROS 2 Jazzy installation documentation](https://docs.ros.org/en/jazzy/Installation.html).
Do not change drivers merely to run the CPU workflow above.

Before the commands below, install Isaac Sim in a **new** host environment named
`env_isaacsim` under the project root, install ROS Jazzy with colcon and RViz,
and restore the scene/asset folders from the portable bundle. The environment
must support the camera script's dependencies, including NumPy and Pillow.
Do not copy the old PC's virtual environments or run the scene builders over
your saved scenes just to start the application. These are launch steps for a
prepared host, not an automated fresh Isaac installation.

### 1. Build the ROS package (prepared host only)

Use a terminal with no unrelated Python environment activated:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-select vla_sim
cd ..
```

### 2. Open the camera scene — terminal A, from the project root

```bash
source /opt/ros/jazzy/setup.bash
./env_isaacsim/bin/python isaac/scripts/wrist_camera_ros.py
```

Wait for `WRIST_CAMERA_READY`. **Do not press Play in this window.** The camera
publishes while the robot stays frozen; pressing Play invalidates that preview's
safety checks and can close it. ROS camera communication uses domain **44**.

### 3. View the images in RViz — terminal B, from the project root

```bash
source /opt/ros/jazzy/setup.bash
ROS_DOMAIN_ID=44 RMW_IMPLEMENTATION=rmw_fastrtps_cpp rviz2 -d configs/wrist_camera.rviz
```

If an Image display is unchecked, enable it in RViz's Displays panel.

### 4. Optional host CPU detector for live ROS images

The offline Docker image intentionally has no ROS/network access. For the
existing live detector, create the separate host CPU environment **only if it
does not already exist**; do not install these packages into Isaac's environment:

```bash
python3.12 -m venv env_yolov9_cpu
./env_yolov9_cpu/bin/python -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
./env_yolov9_cpu/bin/python -m pip install -r configs/requirements-yolov9-cpu.txt
./env_yolov9_cpu/bin/python scripts/fetch_official_yolov9.py
```

With terminal A still publishing, run in another terminal:

```bash
source /opt/ros/jazzy/setup.bash
./env_yolov9_cpu/bin/python scripts/detect_bottle_rgbd.py --detector official-yolov9
```

This requires five stable bottle observations and matching depth/calibration/TF.
It may reject the scene if YOLO cannot confidently recognize the bottle. It does
not command motion. Close the preview window or use Ctrl+C in its terminal when
finished. Close RViz separately.

**Robot motion is not part of this quick start.** Bottle pick-and-place remains
unvalidated. Re-run isolated validation on the destination PC before any motion
execution; never copy old success reports as permission to move.

## Manual labelling with CVAT

CVAT is an independent installation; this project's Compose file does not start
or back up CVAT. If your existing local CVAT is running on port 8080, open
[http://localhost:8080](http://localhost:8080) and sign in directly in the browser.

1. Import your selected PNG images, not metadata/contact sheets.
2. Review duplicates and the [flagged close-up views](tableimage/zoom_views/README.md).
3. Use consistent class names and label all visible instances of those classes.
4. Export annotations in YOLO detection format and retain the class-ID mapping.
5. Save exported labels alongside your dataset or under `data/datasets/`, then
	 create a new transfer bundle. Labels still inside CVAT are not included.

Model training is not configured by this Docker quick start. Validate the manual
labels and prepare independent train/validation data before adding training.

## Common problems

| Problem | What to check |
|---|---|
| Docker connection/permission error | Docker daemon must be running and your user must have access; follow Docker's installation guidance. |
| No images found after cloning | Git excludes images; restore the dataset from the transfer bundle. |
| Output directory already exists | Choose a new `OUTPUT=/output/<new-name>`; previous results are protected. |
| `make doctor` fails to download | Downloads happen during `make build`; check internet access and rebuild. Runtime has no network. |
| Isaac scene assets are missing | Restore assets and portable scene copies; the Orbbec HTTPS asset still requires internet. |
| Camera closes when Play is pressed | Leave Play stopped; images publish automatically in this preview. |
| No bottle detection | Inspect the current image and predictions; do not bypass confidence/safety gates. |

### What goes in Git

- Source, Docker/Compose files, configuration, tests, guides and lightweight dataset metadata.
- **Not** virtual environments, model weights, generated assets/scenes, images,
	annotations, secrets or runtime outputs. Use the transfer bundle for local data.
- Existing local files are preserved; `.gitignore` is not a deletion tool.
- Future changes still need review, a Git commit and an explicit push.

### Useful entry points

- [Portable setup, transfer and Git workflow](docs/portable_setup.md)
- [Manual annotation guide](tableimage/README.md)
- [Close-up image quality notes](tableimage/zoom_views/README.md)
- [Official YOLO model provenance and limitations](docs/official_yolov9.md)
- [Wrist camera: leave Play stopped](docs/wrist_camera.md)

### Licensing

Official YOLOv9 source retains its upstream **GPL-3.0** license. The ROS package's
license does not relicense YOLO, checkpoints or third-party simulation assets.
Isaac Sim and remote NVIDIA assets have their own terms. Review these before
redistributing the Docker image or scene bundle. No repository-wide license is
invented by this setup.
