"""Procedural tabletop props: original geometry, no external asset downloads.

Use only on a fresh/stopped stage, after SimulationApp initialization.
The bottle is a capped rigid visual model, not a liquid simulation.
"""

PROPS_PATH = "/World/TabletopProps"
LAYOUT = {
    "GreenCube": (0.70, 0.14, 0.0),
    "BlackCube": (0.73, -0.07, 0.0),
    "WaterBottle": (0.72, 0.34, 0.0),
    "OrangeCan": (0.93, 0.18, 0.0),
    "YellowBlock": (0.92, -0.16, 0.0),
}


def add_props(stage):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    if stage.GetPrimAtPath(PROPS_PATH):
        raise ValueError("Tabletop props already exist; refusing to overwrite")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1) > 1e-8 or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("Metres and Z up required")
    UsdGeom.Xform.Define(stage, PROPS_PATH)

    def material(name, color, roughness=0.45, metallic=0.0):
        mat = UsdShade.Material.Define(stage, f"{PROPS_PATH}/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat, color

    green = material("Green", (0.03, 0.65, 0.12))
    black = material("Black", (0.018, 0.022, 0.028), 0.32)
    aqua = material("BottlePlastic", (0.30, 0.75, 0.90), 0.18)
    blue = material("BottleLabel", (0.025, 0.20, 0.70))
    white = material("Cap", (0.92, 0.94, 0.96))
    orange = material("CanPaint", (0.95, 0.25, 0.025), 0.3, 0.35)
    silver = material("CanMetal", (0.62, 0.67, 0.72), 0.25, 0.8)
    yellow = material("Yellow", (0.95, 0.72, 0.035))

    def body(name, mass):
        root = UsdGeom.Xform.Define(stage, f"{PROPS_PATH}/{name}")
        UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(*LAYOUT[name]))
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
        root.GetPrim().CreateAttribute("user:objectName", Sdf.ValueTypeNames.String).Set(name)
        return str(root.GetPath())

    def finish(shape, pos, mat, collide=True):
        UsdGeom.XformCommonAPI(shape).SetTranslate(Gf.Vec3d(*pos))
        shape.CreateDisplayColorAttr([Gf.Vec3f(*mat[1])])
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat[0])
        if collide:
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        return shape

    def block(path, pos, size, mat):
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1.0)
        shape.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
        UsdGeom.XformCommonAPI(shape).SetScale(Gf.Vec3f(*size))
        return finish(shape, pos, mat)

    def cylinder(path, z, radius, height, mat, collide=True):
        shape = UsdGeom.Cylinder.Define(stage, path)
        shape.CreateAxisAttr(UsdGeom.Tokens.z)
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(height)
        shape.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2), Gf.Vec3f(radius, radius, height / 2)])
        return finish(shape, (0, 0, z), mat, collide)

    for name, mat in (("GreenCube", green), ("BlackCube", black)):
        path = body(name, 0.05)
        block(f"{path}/Shape", (0, 0, 0.025), (0.05, 0.05, 0.05), mat)
    path = body("YellowBlock", 0.08)
    block(f"{path}/Shape", (0, 0, 0.02), (0.09, 0.045, 0.04), yellow)
    path = body("OrangeCan", 0.12)
    cylinder(f"{path}/Body", 0.055, 0.029, 0.11, orange)
    cylinder(f"{path}/Top", 0.111, 0.029, 0.002, silver)
    path = body("WaterBottle", 0.25)
    cylinder(f"{path}/Body", 0.07, 0.032, 0.14, aqua)
    shoulder = UsdGeom.Sphere.Define(stage, f"{path}/Shoulder")
    # A true sphere keeps rendering and collision geometry consistent in PhysX.
    shoulder.CreateRadiusAttr(0.032)
    shoulder.CreateExtentAttr([Gf.Vec3f(-0.032), Gf.Vec3f(0.032)])
    finish(shoulder, (0, 0, 0.13), aqua)
    cylinder(f"{path}/Neck", 0.17, 0.014, 0.04, aqua)
    cylinder(f"{path}/Cap", 0.196, 0.018, 0.012, white)
    cylinder(f"{path}/Label", 0.077, 0.0325, 0.048, blue, collide=False)
    return [f"{PROPS_PATH}/{name}" for name in LAYOUT]


def check_layout(stage):
    """Static bounds checks only: not a robot trajectory validation."""
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render], False, True)
    bounds = {}
    table = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/table")).ComputeAlignedRange()
    for name in LAYOUT:
        path = f"{PROPS_PATH}/{name}"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Missing rigid prop: {path}")
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = box.GetMin(), box.GetMax()
        if box.IsEmpty() or abs(lo[2] - table.GetMax()[2]) > 0.002:
            raise RuntimeError(f"Prop not on tabletop: {name}")
        if any(lo[i] < table.GetMin()[i] + 0.025 or hi[i] > table.GetMax()[i] - 0.025 for i in (0, 1)):
            raise RuntimeError(f"Prop too close to table edge: {name}")
        bounds[name] = box

    def overlaps(a, b, margin):
        return all(a.GetMin()[i] < b.GetMax()[i] + margin and a.GetMax()[i] > b.GetMin()[i] - margin for i in range(3))

    for name, box in bounds.items():
        for other, other_box in bounds.items():
            if name != other and overlaps(box, other_box, 0.02):
                raise RuntimeError(f"Props overlap: {name}, {other}")
        for path in ("/World/Cube", "/World/PlacementBox"):
            if overlaps(box, cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange(), 0.04):
                raise RuntimeError(f"Prop crowds task object: {name}, {path}")
        robot = stage.GetPrimAtPath("/World/panda")
        for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                robot_box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if robot_box.IsEmpty() or overlaps(box, robot_box, 0.02):
                    raise RuntimeError(f"Cannot establish robot clearance for {name}: {prim.GetPath()}")
    return {name: {"min": list(b.GetMin()), "max": list(b.GetMax())} for name, b in bounds.items()}