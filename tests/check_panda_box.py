"""Isolated no-ROS box test. Drops a TEST cube, never drives the Panda."""
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac/scripts"))

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
try:
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.xforms import get_world_pose
    from pxr import Gf, UsdGeom, UsdPhysics
    from placement_box import BOX_PATH, CENTER_XY, THICKNESS, add_placement_box

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    add_placement_box(stage)
    before = stage.GetRootLayer().ExportToString()
    try:
        add_placement_box(stage)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate box must be rejected")
    assert stage.GetRootLayer().ExportToString() == before
    children = stage.GetPrimAtPath(BOX_PATH).GetChildren()
    assert len(children) == 5
    assert all(p.HasAPI(UsdPhysics.CollisionAPI) and not p.HasAPI(UsdPhysics.RigidBodyAPI) for p in children)
    cube = UsdGeom.Cube.Define(stage, "/World/TestCube")
    cube.CreateSizeAttr(0.05)
    UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(*CENTER_XY, 0.16))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.05)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(240):
        app.update()
    p, _ = get_world_pose("/World/TestCube", fabric=True)
    np.testing.assert_allclose(p, [*CENTER_XY, THICKNESS + 0.025], atol=0.002)
    timeline.stop()
    print(f"BOX_TEST_PASSED: 5 static panels, duplicate rejected, test cube settled inside at {p}", flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close(skip_cleanup=False)