"""Apply reviewed PR #19 appearance without replacing latest physical scenes."""

import hashlib
import json
from pathlib import Path
import shutil
from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdUtils


def physical_signature(stage, ignore_roots=()):
    """Include all physical opinions and each collider's complete subtree."""
    digest = hashlib.sha256()
    roots = []
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if any(path == root or path.startswith(root+'/') for root in ignore_roots):
            continue
        collider = prim.HasAPI(UsdPhysics.CollisionAPI)
        if collider:
            roots.append(path)
        geometry = collider or any(path.startswith(root+'/') for root in roots)
        shape_attrs = {'points','faceVertexCounts','faceVertexIndices','radius','height','size','axis','xformOpOrder'}
        attrs = [(a.GetName(),str(a.Get())) for a in prim.GetAttributes()
                 if a.GetName().startswith(('physics:','physx','drive:'))
                 or (geometry and (a.GetName() in shape_attrs or a.GetName().startswith('xformOp:')))]
        attrs += [(r.GetName(),str(r.GetTargets())) for r in prim.GetRelationships()
                  if r.GetName().startswith('physics:')]
        if attrs:
            digest.update((path+json.dumps(attrs)).encode())
            if geometry and prim.IsA(UsdGeom.Xformable):
                digest.update(str(cache.GetLocalToWorldTransform(prim)).encode())
    return digest.hexdigest()


def apply(stage, bundle, tid, assets_root):
    if tid == 'L2-07':
        return {'revision':'visual_alignment_r3_20261014','applied':False,'reason':'Explicitly excluded by PR #19'}
    before = physical_signature(stage)
    tray_path = '/World/Household/plug_storage_base'
    # The newer complete delivery owns every task prop, including its material
    # and tray. PR #19 is a room/background overlay, not a prop rollback.
    measured_tray = False
    exceptions = (tray_path,) if measured_tray else ()
    unchanged_before = physical_signature(stage,exceptions)
    source_path = bundle / tid / 'static/scene.usdc'
    source = Usd.Stage.Open(str(source_path))
    layer = source.Flatten()
    texture_root = assets_root / 'visual_r3/textures'
    texture_root.mkdir(parents=True,exist_ok=True)

    def portable(path):
        if Path(path).name in {'OmniPBR.mdl','OmniGlass.mdl','OmniSurface.mdl'}:
            return Path(path).name
        file = Path(path)
        if not file.is_absolute():
            file = source_path.parent / file
        if not file.is_file():
            raise FileNotFoundError('R3 visual dependency: '+str(file))
        target = texture_root / file.name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(file.read_bytes()).digest():
                raise ValueError('Conflicting visual texture: '+file.name)
        else:
            shutil.copy2(file,target)
        return '../../visual_r3/textures/'+file.name

    UsdUtils.ModifyAssetPaths(layer,portable)
    revised = Usd.Stage.Open(layer)
    copied = []

    def copy(path):
        path = Sdf.Path(path)
        parent = stage.GetPrimAtPath(path.GetParentPath())
        if not parent or not parent.IsDefined():
            stage.DefinePrim(path.GetParentPath(),'Xform')
        Sdf.CopySpec(layer,path,stage.GetRootLayer(),path)
        copied.append(str(path))
    # Render materials do not carry over collision materials, poses, cameras,
    # rigid/deformable bodies, joints, finger details or older robot meshes.
    for prim in revised.Traverse():
        path = str(prim.GetPath())
        if (not prim.IsA(UsdShade.Material) or path.startswith(('/World/Robots/',
                '/World/Household/', '/World/L2/', '/World/L3Wiping/', '/World/ReviewAssets/'))
                or prim.HasAPI(UsdPhysics.MaterialAPI)):
            continue
        if stage.GetPrimAtPath(path) or path.startswith('/World/Materials/R3'):
            copy(path)
    for path in ('/World/ReferenceDetail/TableSurface','/World/VisualRevision'):
        if revised.GetPrimAtPath(path):
            copy(path)
    if measured_tray:
        if not revised.GetPrimAtPath(tray_path):
            raise ValueError('PR #19 measured plug tray is absent')
        copy(tray_path)
    for prim in revised.Traverse():
        if prim.HasAPI(UsdLux.LightAPI):
            target = stage.GetPrimAtPath(prim.GetPath())
            if not target:
                continue
            for name in ('inputs:intensity','inputs:color'):
                attr = prim.GetAttribute(name)
                if attr:
                    target.GetAttribute(name).Set(attr.Get())
    after = physical_signature(stage)
    if unchanged_before != physical_signature(stage,exceptions):
        raise ValueError('Visual overlay changed physical geometry or properties: '+tid)
    result = {'revision':'visual_alignment_r3_20261014','applied':True,
            'pull_request_head':'11472ffa6d139e11e8b78c04381795da7297d706',
            'source_static_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'copied_render_prims':copied,'physical_signature_before':before,
            'physical_signature_after':after,'latest_physics_and_poses_preserved':not measured_tray,
            'unchanged_physics_signature':unchanged_before,
            'physical_changes':([{'root':tray_path,'reason':'PR #19 user-measured 92.5 mm outer tray / 50 mm height supersedes 101.4 mm / 55 mm estimate; original placement retained'}] if measured_tray else []),
            'excluded_changes':['older task prop geometry and materials, including the PR #19 tray','older robot geometry and poses','axis-aligned replacement of the current rotated target','leaf silhouette changes without matching collider updates']}
    if measured_tray:
        from video_scene_metadata import record
        result['effective_tray_record'] = record(stage,tray_path)
    return result
