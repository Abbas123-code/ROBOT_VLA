import argparse
import math
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "isaac/assets/bullet3/examples/pybullet/gym/pybullet_data"
GENERATED = ROOT / "isaac/assets/generated"
HOME = (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4)


def prepare_urdf(source, name):
    import trimesh

    GENERATED.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    tree.getroot().insert(0, ET.Comment("Modified by ROBOT_VLA: absolute mesh paths and simplified link6 visual."))
    for mesh in tree.iter("mesh"):
        path = source.parent / mesh.attrib["filename"].removeprefix("package://")
        # STL conversion removes unused OBJ vertices rejected by Isaac's mesh validator.
        if name == "panda" and path.name == "link6.obj":
            shape = trimesh.load_mesh(source.parent / "meshes/collision/link6.obj", process=True)
            shape.remove_unreferenced_vertices()
            path = GENERATED / "link6.stl"
            shape.export(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path))
    GENERATED.mkdir(parents=True, exist_ok=True)
    output = GENERATED / f"{name}.urdf"
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def build_scene(output):
    import omni.graph.core as og
    import omni.usd
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

    converted = {}
    for name, source in (("panda", DATA / "franka_panda/panda.urdf"),
                         ("table", DATA / "table/table.urdf")):
        config = URDFImporterConfig()
        config.urdf_path = str(prepare_urdf(source, name))
        config.usd_path = str(GENERATED / name)
        config.fix_base = True
        config.allow_self_collision = False
        importer = URDFImporter()
        importer.config = config
        converted[name] = importer.import_urdf()

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(60)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

    for name, path in converted.items():
        prim = stage.DefinePrim(f"/World/{name}", "Xform")
        prim.GetReferences().AddReference(str(path))
        prim.GetVariantSet("Physics").SetVariantSelection("physx")

    table = stage.GetPrimAtPath("/World/table")
    UsdGeom.XformCommonAPI(table).SetTranslate(Gf.Vec3d(0.35, 0, -0.625))
    # The table is scenery, not a massless robot articulation.
    for prim in list(Usd.PrimRange(table)):
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    robot = stage.GetPrimAtPath("/World/panda")
    roots = []
    for prim in Usd.PrimRange(robot):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(prim.GetPath())
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName().startswith("panda_joint"):
            index = int(prim.GetName().removeprefix("panda_joint")) - 1
            position = math.degrees(HOME[index])
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(400.0)
            drive.CreateDampingAttr(40.0)
            drive.CreateMaxForceAttr(87.0 if index < 4 else 12.0)
            drive.CreateTargetPositionAttr(position)
            state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
            state.CreatePositionAttr(position)
            state.CreateVelocityAttr(0.0)
    if len(roots) != 1:
        raise RuntimeError(f"Expected one Panda articulation root, found {roots}")

    def cube(path, position, scale, color, dynamic=False):
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1.0)
        shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        transform = UsdGeom.XformCommonAPI(shape)
        transform.SetTranslate(Gf.Vec3d(*position))
        transform.SetScale(Gf.Vec3f(*scale))
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        if dynamic:
            UsdPhysics.RigidBodyAPI.Apply(shape.GetPrim())
            UsdPhysics.MassAPI.Apply(shape.GetPrim()).CreateMassAttr(0.05)

    cube("/World/Floor", (0, 0, -0.68), (6, 6, 0.1), (0.22, 0.25, 0.28))
    cube("/World/Cube", (0.5, 0, 0.04), (0.05, 0.05, 0.05), (0.85, 0.12, 0.07), True)
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000)
    camera = UsdGeom.Camera.Define(stage, "/World/Overview")
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(2, 2, 1.6), Gf.Vec3d(0.3, 0, 0.2), Gf.Vec3d(0, 0, 1))
    camera.AddTransformOp().Set(view.GetInverse())

    keys = og.Controller.Keys
    og.Controller.edit({"graph_path": "/World/ROS2", "evaluator_name": "execution"}, {
        keys.CREATE_NODES: [
            ("Tick", "omni.graph.action.OnPlaybackTick"),
            ("Time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("States", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("Commands", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("Control", "isaacsim.core.nodes.IsaacArticulationController"),
            ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        keys.CONNECT: [
            ("Tick.outputs:tick", "States.inputs:execIn"),
            ("Tick.outputs:tick", "Commands.inputs:execIn"),
            ("Commands.outputs:execOut", "Control.inputs:execIn"),
            ("Tick.outputs:tick", "Clock.inputs:execIn"),
            ("Time.outputs:simulationTime", "States.inputs:timeStamp"),
            ("Time.outputs:simulationTime", "Clock.inputs:timeStamp"),
            ("Context.outputs:context", "States.inputs:context"),
            ("Context.outputs:context", "Commands.inputs:context"),
            ("Context.outputs:context", "Clock.inputs:context"),
            ("Commands.outputs:jointNames", "Control.inputs:jointNames"),
            ("Commands.outputs:positionCommand", "Control.inputs:positionCommand"),
        ],
        keys.SET_VALUES: [
            ("Context.inputs:domain_id", 42),
            ("Context.inputs:useDomainIDEnvVar", False),
            ("Time.inputs:resetOnStop", True),
            ("States.inputs:targetPrim", [roots[0]]),
            ("Control.inputs:targetPrim", [roots[0]]),
            ("States.inputs:topicName", "/vla_sim/joint_states"),
            ("Commands.inputs:topicName", "/vla_sim/joint_command"),
            ("Clock.inputs:topicName", "/clock"),
        ],
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(output)):
        raise RuntimeError(f"Could not export {output}")
    print(f"SCENE_SAVED={output}", flush=True)
    print(f"ARTICULATION_ROOT={roots[0]}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="0 keeps the simulation running")
    parser.add_argument("--output", type=Path, default=ROOT / "isaac/scenes/panda_tabletop.usda")
    args = parser.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.asset.importer.urdf")
        enable_extension("isaacsim.ros2.bridge")
        app.update()
        build_scene(args.output.resolve())
        if args.build_only:
            return
        import omni.timeline
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera("/World/Overview")
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        step = 0
        while app.is_running() and (args.steps <= 0 or step < args.steps):
            start = time.monotonic()
            app.update()
            step += 1
            time.sleep(max(0.0, 1 / 60 - (time.monotonic() - start)))
        timeline.stop()
        for _ in range(10):
            app.update()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close(skip_cleanup=False)


if __name__ == "__main__":
    main()