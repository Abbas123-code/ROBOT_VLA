"""Create a separate stopped Panda scene with a receiving box; no robot motion.

Run using env_isaacsim/bin/python. The original scene is not modified.
The new scene's ROS graph is disabled so opening/playing it cannot consume
commands from an old domain-42 publisher. Enable it only after a new preflight.
"""
import hashlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "isaac/scenes/panda_tabletop.usda"
OUTPUT = ROOT / "isaac/scenes/panda_tabletop_with_box.usda"


def build():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from placement_box import BOX_PATH, CENTER_XY, HEIGHT, OUTER_SIZE, add_placement_box

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing scene: {OUTPUT}")
    original_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    baseline = Usd.Stage.Open(str(SOURCE))
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths.append(str(SOURCE))
    # Stage-level units/up-axis are not inherited from a sublayer.
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(baseline))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(baseline))
    stage.SetTimeCodesPerSecond(baseline.GetTimeCodesPerSecond())
    # Imported collision meshes use purpose="guide" and may be hidden.
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
        False, True,
    )
    low = Gf.Vec3d(CENTER_XY[0] - OUTER_SIZE[0] / 2, CENTER_XY[1] - OUTER_SIZE[1] / 2, 0)
    high = Gf.Vec3d(CENTER_XY[0] + OUTER_SIZE[0] / 2, CENTER_XY[1] + OUTER_SIZE[1] / 2, HEIGHT)
    table = stage.GetPrimAtPath("/World/table")
    robot = stage.GetPrimAtPath("/World/panda")
    cube = stage.GetPrimAtPath("/World/Cube")
    if not all((table, robot, cube)):
        raise RuntimeError("Baseline Panda/table/cube prims are required")
    table_range = cache.ComputeWorldBound(table).ComputeAlignedRange()
    if abs(table_range.GetMax()[2]) > 0.002 or any(
        low[i] < table_range.GetMin()[i] + 0.02 or high[i] > table_range.GetMax()[i] - 0.02
        for i in (0, 1)
    ):
        raise RuntimeError("Box does not fit the expected tabletop at world Z=0")
    # Check the entire new box envelope, not only its walls, against the saved
    # robot/cube geometry. This is a static overlap check, not path planning.
    for root in (robot, cube):
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if bounds.IsEmpty():
                raise RuntimeError(f"Cannot check collider bounds: {prim.GetPath()}")
            if all(bounds.GetMin()[i] < high[i] + 0.01 and bounds.GetMax()[i] > low[i] - 0.01 for i in range(3)):
                raise RuntimeError(f"Box too close to saved collider: {prim.GetPath()}")

    report = add_placement_box(stage)
    graph = stage.GetPrimAtPath("/World/ROS2")
    if graph:
        graph.SetActive(False)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    # Keep the baseline as a sublayer rather than rewriting its contents.
    # Like the imported asset paths, this path must be updated if moved.
    if not stage.GetRootLayer().Export(str(OUTPUT)):
        raise RuntimeError("Scene export failed")
    reopened = Usd.Stage.Open(str(OUTPUT))
    assert reopened and reopened.GetPrimAtPath(BOX_PATH)
    assert not reopened.GetPrimAtPath("/World/ROS2").IsActive()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == original_hash
    report.update(scene=str(OUTPUT), source_scene_unchanged=True, ros_graph_enabled=False)
    print(json.dumps(report, indent=2), flush=True)
    print("BOX_SCENE_CREATED", flush=True)


if __name__ == "__main__":
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        build()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close(skip_cleanup=False)