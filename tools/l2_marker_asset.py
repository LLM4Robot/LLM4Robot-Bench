"""Install the user-supplied marker exterior on the L2-01 detachable bodies.

The input is a closed, single-body visual reconstruction. Its exterior geometry
matches the fitted scene exactly; it supplies the new label/materials without
replacing the hollow cap by the archive's solid closed-cap collider.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'configs/task_references/l2_marker_asset_20261016.json'
STAMP = 'gptBenchmark:markerAssetOverride'


def repair_label_visibility(stage, result):
    # RTX's translucent substrate pass covers the opaque printed sleeve even
    # though the sleeve is outside it. An opaque substrate renders the existing
    # sleeve correctly; no geometry, UV, collision or pose changes are needed.
    surface = stage.GetPrimAtPath('/World/L2/marker/LatestLooks/SmokeClear/Surface')
    surface.GetAttribute('inputs:opacity').Set(1.0)
    result['render_adaptation'] = {
        'revision': 'opaque-substrate-label-visibility-v1',
        'substrate_opacity': 1.0,
        'source_substrate_opacity': 0.34,
        'reason': 'Prevent the translucent substrate from obscuring the exterior printed sleeve in RTX',
        'geometry_uv_and_physics_unchanged': True,
    }
    stage.GetPrimAtPath('/World/L2/marker').CreateAttribute(
        STAMP, Sdf.ValueTypeNames.String, custom=True).Set(json.dumps(result, sort_keys=True))
    return result


def apply(stage):
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    asset = ROOT / 'assets/tuojing' / reference['asset_path']
    if hashlib.sha256(asset.read_bytes()).hexdigest() != reference['asset_sha256']:
        raise ValueError('Supplied L2-01 marker archive differs from its lock')
    native = '/World/L2/marker'
    root = stage.GetPrimAtPath(native)
    stamp = root.GetAttribute(STAMP)
    if stamp and stamp.Get():
        result = json.loads(stamp.Get())
        if result['revision'] != reference['revision']:
            raise ValueError('Rebuild L2-01 before changing its marker override')
        return repair_label_visibility(stage, result)
    source = Usd.Stage.Open(str(asset))
    layer = source.Flatten()
    parts = {'Body': [], 'Cap': []}
    for prim in source.GetPrimAtPath('/Marker/Body/Visuals').GetChildren():
        name = prim.GetName()
        body = 'Cap' if name.startswith('Cap') else 'Body'
        target = f'{native}/{body}/LatestVisuals/{name}'
        current = stage.GetPrimAtPath(target)
        if not prim.IsA(UsdGeom.Mesh) or not current.IsA(UsdGeom.Mesh):
            raise ValueError('Unexpected marker exterior: ' + name)
        # Protect the already verified hollow-cap fit. A different exterior
        # shape must get an explicit geometric adaptation, not a silent swap.
        for attribute in ('points', 'faceVertexCounts', 'faceVertexIndices'):
            if not np.array_equal(prim.GetAttribute(attribute).Get(), current.GetAttribute(attribute).Get()):
                raise ValueError('New marker geometry needs a new fit: ' + name)
        parts[body].append(name)
    for body, names in parts.items():
        if set(names) != {p.GetName() for p in stage.GetPrimAtPath(f'{native}/{body}/LatestVisuals').GetChildren()}:
            raise ValueError('Marker exterior part list changed')

    old_looks = Sdf.Path('/Marker/Body/Looks')
    new_looks = Sdf.Path(native + '/LatestLooks')
    stage.RemovePrim(new_looks)
    Sdf.CopySpec(layer, old_looks, stage.GetRootLayer(), new_looks)
    copied_roots = [str(new_looks)]
    for body, names in parts.items():
        for name in names:
            target = f'{native}/{body}/LatestVisuals/{name}'
            stage.RemovePrim(target)
            Sdf.CopySpec(layer, '/Marker/Body/Visuals/' + name, stage.GetRootLayer(), target)
            copied_roots.append(target)
    # Materials live outside each copied mesh's subtree. Redirect every binding
    # and shader connection to the newly installed material namespace.
    for path in copied_roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            for rel in prim.GetRelationships():
                targets = rel.GetTargets()
                if targets:
                    rel.SetTargets([p.ReplacePrefix(old_looks, new_looks) for p in targets])
            for attr in prim.GetAttributes():
                connections = attr.GetConnections()
                if connections:
                    attr.SetConnections([p.ReplacePrefix(old_looks, new_looks) for p in connections])

    result = {**reference, 'exterior_parts': parts,
              'exterior_geometry_exactly_matches_previous': True,
              'native_hollow_cap_nib_springs_masses_and_collisions_preserved': True,
              'body_and_cap_remain_separate_dynamic_bodies': True}
    return repair_label_visibility(stage, result)
