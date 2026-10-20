"""Static, open-top receiving box for the Panda table (metres, Z up).

Import inside Isaac Sim after SimulationApp starts. This module never starts
physics, moves the robot/cube, changes drives, or saves a stage.
"""

BOX_PATH = "/World/PlacementBox"
CENTER_XY = (0.45, -0.25)
OUTER_SIZE = (0.20, 0.20)
HEIGHT = 0.06
THICKNESS = 0.008
TABLE_TOP = 0.0


def parts():
    """Five non-overlapping solid panels; no lid and no filled interior."""
    x, y = OUTER_SIZE
    t, h = THICKNESS, HEIGHT
    return (
        ("Bottom", (0, 0, t / 2), (x, y, t)),
        ("WallXMinus", (-(x - t) / 2, 0, (h + t) / 2), (t, y, h - t)),
        ("WallXPlus", ((x - t) / 2, 0, (h + t) / 2), (t, y, h - t)),
        ("WallYMinus", (0, -(y - t) / 2, (h + t) / 2), (x - 2 * t, t, h - t)),
        ("WallYPlus", (0, (y - t) / 2, (h + t) / 2), (x - 2 * t, t, h - t)),
    )


def add_placement_box(stage):
    """Author five static colliders; caller must use a stopped/fresh stage."""
    from pxr import Gf, UsdGeom, UsdPhysics

    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-8:
        raise ValueError("Placement box requires metre stage units")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("Placement box requires Z up")
    if stage.GetPrimAtPath(BOX_PATH):
        raise ValueError("PlacementBox already exists; refusing to replace it")
    world = stage.GetPrimAtPath("/World")
    if not world or UsdGeom.Xformable(world).ComputeLocalToWorldTransform(0) != Gf.Matrix4d(1):
        raise ValueError("Expected an untransformed /World")

    box = UsdGeom.Xform.Define(stage, BOX_PATH)
    UsdGeom.XformCommonAPI(box).SetTranslate(Gf.Vec3d(*CENTER_XY, TABLE_TOP))
    box.GetPrim().SetDocumentation(
        "Static open-top receiving box. Robot placement path not yet validated."
    )
    for name, center, dimensions in parts():
        panel = UsdGeom.Cube.Define(stage, f"{BOX_PATH}/{name}")
        panel.CreateSizeAttr(1.0)
        panel.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
        panel.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.42, 0.68)])
        transform = UsdGeom.XformCommonAPI(panel)
        transform.SetTranslate(Gf.Vec3d(*center))
        transform.SetScale(Gf.Vec3f(*dimensions))
        UsdPhysics.CollisionAPI.Apply(panel.GetPrim())
        # No RigidBodyAPI: the panels are fixed scenery, not loose walls.

    return {
        "box_path": BOX_PATH,
        "center_xy_metres": list(CENTER_XY),
        "outer_size_metres": [*OUTER_SIZE, HEIGHT],
        "inner_opening_metres": [v - 2 * THICKNESS for v in OUTER_SIZE],
        "inside_floor_world_z": TABLE_TOP + THICKNESS,
        "rim_world_z": TABLE_TOP + HEIGHT,
        "resting_cube_center_world_xyz": [*CENTER_XY, TABLE_TOP + THICKNESS + 0.025],
        "note": "Geometric target only. No robot trajectory or release validated.",
    }