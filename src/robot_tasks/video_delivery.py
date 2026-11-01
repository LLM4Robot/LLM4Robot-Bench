"""Complete, hash-checked task scenes from the 2026-10-15 delivery.

These scenes supply the room, table, robot geometry, optics, props and initial
placements together. Older individual-prop bundles do not enter this path.
"""

from dataclasses import replace
from functools import lru_cache
import hashlib
import json

from .paths import PROJECT_ROOT

DELIVERY = "video_20261015"
MANIFEST = PROJECT_ROOT / "assets/tuojing/video_20261015.json"
VISUAL_SETUP = PROJECT_ROOT / "configs/task_references/video_visual_setup_20261015.json"


@lru_cache(maxsize=1)
def _visual_setup_record():
    """Visible video settings, estimated in a physically valid robot model."""
    contents = VISUAL_SETUP.read_bytes()
    return json.loads(contents), hashlib.sha256(contents).hexdigest()


def visual_setup():
    return _visual_setup_record()[0]


def setup_for(task_id):
    setup = visual_setup()['tasks'][task_id]
    if setup['runtime_scene_sha256'] != scenes()[task_id]['runtime_sha256']:
        raise ValueError('Visual initial setup refers to another task scene: ' + task_id)
    return setup


def setup_digest():
    return _visual_setup_record()[1]


def _recorded_task_setup(task_id, recorded_digest):
    if recorded_digest == setup_digest():
        return setup_for(task_id)
    if (not isinstance(recorded_digest, str) or len(recorded_digest) != 64
            or any(c not in '0123456789abcdef' for c in recorded_digest)):
        return None
    path = VISUAL_SETUP.parent / 'video_visual_setup_history' / (recorded_digest + '.json')
    if not path.is_file():
        return None
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != recorded_digest:
        raise ValueError('Modified historical visual setup: ' + str(path))
    record = json.loads(contents)
    from .task_numbering import (
        SCHEME, PREVIOUS_SCHEME, RECLASSIFIED_SCHEME,
        historical_setup_record, previous_task_id, to_legacy,
    )
    if record.get('task_numbering') == SCHEME:
        return record.get('tasks', {}).get(task_id)
    scheme = record.get('task_numbering')
    if scheme in (PREVIOUS_SCHEME, RECLASSIFIED_SCHEME):
        return historical_setup_record(
            record.get('tasks', {}).get(previous_task_id(task_id, scheme)), scheme)
    return historical_setup_record(record.get('tasks', {}).get(to_legacy(task_id)))


def setup_compatible(task_id, recorded_digest):
    """Reuse images/evidence only when the complete task record is unchanged.

    Historical bytes must match their recorded SHA before comparing settings.
    A correction to another task does not invalidate identical scene inputs.
    """
    return _recorded_task_setup(task_id, recorded_digest) == setup_for(task_id)


def planning_setup_compatible(task_id, recorded_digest):
    """Reuse planning checks across light/material-color-only changes.

    This never validates old images or recordings. Joints, camera transforms,
    fixture/background transforms and runtime hashes must still be identical.
    """
    recorded = _recorded_task_setup(task_id, recorded_digest)
    if recorded is None:
        return False
    return ({k: v for k, v in recorded.items() if k != 'render_appearance'}
            == {k: v for k, v in setup_for(task_id).items() if k != 'render_appearance'})


def video_layout(task_id):
    base = task_id.removesuffix('_v3')
    reference_id = 'L2-04' if base == 'L2-06' else base
    reference = json.loads((PROJECT_ROOT / 'configs/task_references/view_pose_landmarks_20261015.json').read_text(encoding='utf-8'))['tasks'][reference_id]
    return {'resolution': reference['source_resolution'], 'rois': reference['rois'],
            'layout_reference_task': reference_id,
            'source_video_sha256': reference['source_video_sha256'] if reference_id == base else None}


def robot_profile(profile, task_id):
    setup = setup_for(task_id)
    limits = {limit.joint: limit for limit in profile.operating_limits}
    mounts = []
    for mount in profile.mounts:
        home = tuple(setup['arms'][mount.name]['home_positions_rad'])
        from .robots.transforms import finite_vector
        finite_vector(home, 6, 'Video initial joints')
        if any(not limits[n].lower <= q <= limits[n].upper for n,q in zip(profile.joint_names,home)):
            raise ValueError('Video initial joint pose exceeds the configured limits')
        mounts.append(replace(mount, home_positions_rad=home))
    return replace(profile, mounts=tuple(mounts))


def initial_gripper_width(config, arm, fallback):
    if not enabled(config):
        return fallback
    width = setup_for(config.scene_id)['arms'][arm]['gripper_width_m']
    from .robots.transforms import finite_vector
    finite_vector([width], 1, 'Video initial gripper width')
    if not config.robot.gripper.width_limits_m[0] <= width <= config.robot.gripper.width_limits_m[1]:
        raise ValueError('Video initial gripper opening exceeds configured limits')
    return width


def effective_cameras(task_id):
    setup = setup_for(task_id)
    reference_id = visual_setup()['camera_reference_task']
    reference_setup = setup_for(reference_id)
    reference_scene = scenes()[reference_id]
    cameras = {}
    for name, row in scenes()[task_id]['cameras'].items():
        current = {**row, **setup['cameras'][name]}
        reference = {**reference_scene['cameras'][name], **reference_setup['cameras'][name]}
        # A task may bind its own USD prim, but cannot override the shared rig.
        # Wrist transforms are relative to link6, not frozen in world space.
        for key in ('parent', 'pose', 'intrinsics', 'resolution', 'clipping_m'):
            if current[key] != reference[key]:
                raise ValueError(f'{task_id} {name}: {key} must match shared camera reference {reference_id}')
        cameras[name] = current
    return cameras


@lru_cache(maxsize=1)
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def scenes():
    return manifest()["tasks"]


def enabled(config):
    return config.raw.get("task", {}).get("scene_delivery") == DELIVERY


def verified_scene(task_id):
    row = scenes()[task_id]
    for appearance in row.get('appearance_overrides', {}).values():
        texture = PROJECT_ROOT / 'assets/tuojing' / appearance['texture']
        if not texture.is_file() or texture.stat().st_size != appearance['bytes']:
            raise FileNotFoundError('Missing task appearance texture: ' + str(texture))
        with texture.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != appearance['sha256']:
                raise ValueError('Modified task appearance texture: ' + str(texture))
    for record in row.get('runtime_dependencies', []):
        dependency = PROJECT_ROOT / "assets/tuojing" / record["path"]
        if (not dependency.is_file() or dependency.stat().st_size != record["bytes"]
                or hashlib.sha256(dependency.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError("Missing or changed runtime dependency: " + str(dependency))
    if row.get('visual_overlay',{}).get('applied'):
        textures = manifest().get('visual_textures',{})
        if not textures:
            raise ValueError('PR #19 visual dependencies are not locked')
        for name, record in textures.items():
            file = PROJECT_ROOT / 'assets/tuojing' / name
            if not file.is_file() or file.stat().st_size != record['bytes']:
                raise FileNotFoundError('Missing PR #19 visual texture: '+str(file))
            with file.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest() != record['sha256']:
                    raise ValueError('Modified PR #19 visual texture: '+str(file))
    path = PROJECT_ROOT / "assets/tuojing" / row["path"]
    if not path.is_file() or path.stat().st_size != row["bytes"]:
        raise FileNotFoundError(f"Missing latest scene: {path}; run tools/install_video_scenes.py")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != row["sha256"]:
        raise ValueError("Modified latest scene: " + str(path))
    contents = row.get('derived_contents')
    if contents:
        dependency = PROJECT_ROOT/'assets/tuojing'/contents['source_path']
        if not dependency.is_file() or dependency.stat().st_size != contents['source_bytes']:
            raise FileNotFoundError('Missing current content asset: '+str(dependency))
        with dependency.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != contents['source_sha256']:
                raise ValueError('Modified current content asset: '+str(dependency))
    runtime = PROJECT_ROOT / "assets/tuojing" / row["runtime_path"]
    if not runtime.is_file() or runtime.stat().st_size != row["runtime_bytes"]:
        raise FileNotFoundError("Missing runtime layer: run tools/prepare_video_scenes.py")
    with runtime.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != row["runtime_sha256"]:
        raise ValueError("Modified runtime scene: " + str(runtime))
    return runtime


def camera_profile(profile, task_id):
    """Bind the common L3-01 rig to each task's existing camera prims."""
    scene = scenes()[task_id]
    cameras = effective_cameras(task_id)
    result = []
    for old in profile.cameras:
        row = cameras[old.name]
        result.append(replace(old, optical_pose_wxyz=tuple(row["pose"]),
                              intrinsics=tuple(row["intrinsics"]),
                              resolution=tuple(row["resolution"]),
                              clipping_m=tuple(row["clipping_m"]),
                              existing_prim_path=row["path"],
                              asset=PROJECT_ROOT / "assets/tuojing" / scene["path"],
                              pose_source="Shared " + visual_setup()['camera_reference_task'] + " camera rig; " + visual_setup()['revision'],
                              intrinsics_source="Shared bottle-cap-task projection; video-derived estimate, not hardware calibration"))
    return replace(profile, cameras=tuple(result),
                   notes="All current video tasks use the L3-01 camera rig: a fixed world overview and fixed link6-relative wrist mounts, with shared lens parameters. Initial robot poses remain task-specific. Image-derived estimates, not hardware calibration.")


def apply_visual_cameras(stage, task_id):
    import numpy as np
    from pxr import Gf, UsdGeom
    from .cameras.isaac_rig import _set_pose
    from .robots.transforms import pose_transform, transform_pose

    changes = {}
    cache = UsdGeom.XformCache()
    for name,row in effective_cameras(task_id).items():
        prim = stage.GetPrimAtPath(row['path'])
        parent_path = '/World' if row['parent']=='world' else scenes()[task_id]['robots'][row['parent']]['path']+'/link6'
        world_parent = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path))).T
        world_immediate = np.array(cache.GetLocalToWorldTransform(prim.GetParent())).T
        local = np.linalg.inv(world_immediate) @ world_parent @ pose_transform(row['pose'])
        _set_pose(prim, transform_pose(local))
        cam = UsdGeom.Camera(prim)
        w,h = row['resolution'];fx,fy,cx,cy = row['intrinsics']
        f = cam.GetFocalLengthAttr().Get();ha,va = w*f/fx,h*f/fy
        cam.GetHorizontalApertureAttr().Set(ha);cam.GetVerticalApertureAttr().Set(va)
        cam.GetHorizontalApertureOffsetAttr().Set((.5-cx/w)*ha)
        cam.GetVerticalApertureOffsetAttr().Set((cy/h-.5)*va)
        changes[name] = {'pose':row['pose'],'intrinsics':row['intrinsics'],'layer':'session'}
    return changes


def apply_visual_background(stage, task_id):
    """Use the reference view's stand visibility and wall-strip placement."""
    import numpy as np
    from pxr import Gf, UsdGeom

    setup = setup_for(task_id)
    changes = {}
    for path, value in setup.get('static_visibility', {}).items():
        prim = stage.GetPrimAtPath(path)
        if not prim or value not in ('inherited', 'invisible'):
            raise ValueError('Invalid shared-view visibility: ' + path)
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(value)
        changes[path] = 'shared L3-01 visibility: ' + value
    reference = setup.get('background_reference')
    if not reference:
        return changes
    shared = setup_for(visual_setup()['camera_reference_task'])['background_reference']
    if reference['world_transform'] != shared['world_transform']:
        raise ValueError('Background placement must match the shared camera reference: ' + task_id)
    matrix = np.array(reference['world_transform'])
    cache = UsdGeom.XformCache()
    for path in reference['paths']:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError('Missing video background feature: ' + path)
        world = np.array(cache.GetLocalToWorldTransform(prim)).T
        parent = np.array(cache.GetLocalToWorldTransform(prim.GetParent())).T
        local = np.linalg.inv(parent) @ matrix @ world
        transform = UsdGeom.Xformable(prim)
        transform.ClearXformOpOrder()
        transform.AddTransformOp().Set(Gf.Matrix4d(local.T))
        changes[path] = 'shared L3-01 wall reference on session layer'
    return changes


def configure_stage(stage, task_id):
    from pxr import UsdGeom, UsdShade, Sdf
    from .backends.tuojing_lab import configure_stage as configure_lab
    base_task_id = task_id.removesuffix('_v3')

    stage.SetEditTarget(stage.GetSessionLayer())
    # The exporter lost the source stage metadata. Its coordinates and physics
    # data are metre/Z-up (0.88 m table, 0.14 m robot TCP), not centimetre/Y-up.
    # Restore metadata only; never rotate or scale the complete geometry.
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    # Built-in MDL modules are provided by Isaac, not by a USDZ's directory.
    # Reauthor only these search-path references on the anonymous session layer.
    mdl = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        attr = prim.GetAttribute("info:mdl:sourceAsset")
        value = attr.Get() if attr else None
        basename = value.path.split("/")[-1].rstrip("]") if isinstance(value, Sdf.AssetPath) else ""
        if basename in {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"}:
            import carb.tokens
            module = carb.tokens.get_tokens_interface().resolve("${kit}/mdl/core/Base/" + basename)
            attr.Set(Sdf.AssetPath(module))
            mdl.append(str(prim.GetPath()))
    report = configure_lab(stage, None)
    camera_changes = apply_visual_cameras(stage, task_id)
    background_changes = apply_visual_background(stage, task_id)
    from .video_appearance import apply as apply_appearance
    appearance_changes = apply_appearance(stage, setup_for(task_id).get('render_appearance'))
    if base_task_id in ('L1-01','L1-05','L2-03','L3-01'):
        import carb
        epsilon = .00001 if base_task_id == 'L1-01' else .0001
        carb.settings.get_settings().set_float('/rtx/translucency/worldEps',epsilon)
        report['translucency_world_epsilon_m'] = epsilon
    if base_task_id == 'L3-01':
        # The bottle has both inner and outer PET faces. RTX realtime's
        # transmission approximation makes its shoulder/neck look metallic.
        # Use bounded path tracing for this task's camera observations as well
        # as its GUI; preserve all geometry and physical properties.
        import carb
        settings = carb.settings.get_settings()
        settings.set_string('/rtx/rendermode', 'PathTracing')
        for name, value in {'spp': 16, 'totalSpp': 16, 'clampSpp': 16,
                            'maxBounces': 12,
                            'maxSpecularAndTransmissionBounces': 16}.items():
            settings.set_int('/rtx/pathtracing/' + name, value)
        report['bottle_transparency_renderer'] = {
            'mode': 'PathTracing', 'samples_per_pixel': 16,
            'max_bounces': 12, 'max_specular_transmission_bounces': 16,
        }
    if base_task_id == 'L3-04':
        from .tasks.bench.l3_sweep_physics import apply as apply_sweep_box
        report['l3_05_box_collision'] = apply_sweep_box(stage)
        from .tasks.bench.l3_brush_handle import apply as apply_brush_handle
        report['l3_05_handle_collision'] = apply_brush_handle(stage)
    report.update(profile='complete_scene_20261015',delivery=DELIVERY, task_id=base_task_id,
                  scene_id=task_id, table_surface_z_m=0.88,
                  arm_spacing_m=0.47, scene_sha256=scenes()[task_id]["sha256"],
                  units_metadata_adapter={"source": scenes()[task_id]["source_units_metadata"],
                                          "effective": {"up": "Z", "meters_per_unit": 1.0},
                                          "geometry_scaled_or_rotated": False},
                  camera_mount_overrides=camera_changes,
                  video_background_overrides=background_changes,
                  video_appearance_overrides=appearance_changes,
                  visual_initial_setup={'revision':visual_setup()['revision'], 'sha256':setup_digest(),
                                        'arms':setup_for(task_id)['arms'], 'basis':'visible real-video settings'},
                  built_in_mdl_session_references=mdl,
                  limitations="Latest complete video-guided reconstruction; unmeasured geometry, illumination and material parameters remain estimates.")
    return report


PROMPT_FRAME = (
    "Task left/right names use the ROBOT'S perspective: the task's LEFT arm is "
    "on overview-image RIGHT and uses tool arm='right'; the task's RIGHT arm is "
    "on overview-image LEFT and uses tool arm='left'. Cup left/right names use "
    "the same robot perspective. Tool IDs stay tied to their bases when crossing "
    "the image centre. Convert task names to tool IDs once; do not mirror coordinates."
)


def reconcile_render_graph():
    """Register Kit's existing viewport pipeline before attaching RGB sensors.

    Opening a saved editor scene can create a valid SDGPipeline in Kit before
    SyntheticData's Python registry sees it. Reuse the actual wrapper node;
    attempting to create another graph at the same path gives an invalid handle.
    """
    import omni.graph.core as og
    import omni.syntheticdata as sd

    interface = sd.SyntheticData.Get()
    adopted = []
    for graph in og.get_global_orchestration_graphs():
        for node in graph.get_nodes():
            try:
                wrapped = node.get_wrapped_graph()
                if not wrapped.is_valid():
                    continue
                path = wrapped.get_path_to_graph()
            except (ValueError, RuntimeError):
                continue
            if path.startswith("/Render/") and path.endswith("/SDGPipeline") and path not in interface._nodeGraphs:
                interface._nodeGraphs[path] = node
                adopted.append(path)
    return adopted
