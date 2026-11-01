"""Reference-photo-inspired L1-01 mug, preserving all heights and scene scale.

Dimensions below are unscaled mesh coordinates, not measured real-cup sizes.
The delivered parent scale is retained (90 mm mesh -> 85 mm in the scene).
"""
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

ROOT = '/World/Household/Fixture'
Z = np.array([0, 2, 3, 7, 9, 15, 27, 42, 50, 60, 72, 86, 89, 90]) / 1000
# Radial displacement shared by inner/outer surfaces and wall colliders.
# Inner profile remains concave in height, as required by cavity scoring.
# Previous screenshot-based profile, retained solely to migrate existing
# runtime assets without applying a second deformation cumulatively.
PREVIOUS_DELTA = .4 * np.array([0, 0, 0, 0, 0, 2.5, 4.5, 4, 3.5, 2.3, .2, -3.8, -5, -5]) / 1000
# Three close-up photographs: a modest lower-middle belly, less pinched mouth.
# These are visual estimates, not new measured dimensions. Shared offsets
# keep the cavity and visible ceramic wall in agreement.
DELTA = np.array([0, 0, .4, .8, .9, 1.2, 1.4, 1.2, 1, .6, .1, -.8, -1, -1]) / 1000
HANDLE_SHIFT = .0012
REVISION = 'real_cup_closeups_v3'


def apply_inner_glaze(stage):
    """Smooth ivory ceramic, without changing scene lighting or using emission."""
    surface = stage.GetPrimAtPath(ROOT + '/Looks/WhiteGlaze/Surface')
    if not surface:
        raise ValueError('Missing mug inner ceramic shader')
    rgb = (.94, .915, .85)  # Ivory albedo; do not bake warm photographic shadows.
    surface.GetAttribute('inputs:diffuseColor').Set(Gf.Vec3f(*rgb))
    surface.GetAttribute('inputs:roughness').Set(.22)
    return {'revision': 'ivory_inner_glaze_v2', 'linear_rgb': list(rgb),
            'scope': 'Inside and rounded WhiteLip share smooth non-emissive glaze'}


def apply_outer_glaze(stage):
    """Reuse the delivered grain maps, with neutral satin-black ceramic response."""
    for name in ['MatteBlack', 'BlackGrain']:
        surface = stage.GetPrimAtPath(ROOT + '/Looks/' + name + '/Surface')
        surface.GetAttribute('inputs:diffuseColor').Set(Gf.Vec3f(.018, .0185, .019))
        surface.GetAttribute('inputs:roughness').Set(.43)
    rough = stage.GetPrimAtPath(ROOT + '/Looks/BlackGrain/roughness')
    # Retain fine spatial variation while avoiding a polished-plastic response.
    rough.CreateAttribute('inputs:scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.35,.35,.35,1))
    rough.CreateAttribute('inputs:bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.27,.27,.27,0))
    grain = ROOT + '/Looks/BlackGrain'
    uv = UsdShade.Shader.Define(stage, grain + '/FineGrainUV')
    uv.CreateIdAttr('UsdTransform2d')
    uv.CreateInput('in', Sdf.ValueTypeNames.Float2).ConnectToSource(
        UsdShade.Shader(stage.GetPrimAtPath(grain + '/UV')).ConnectableAPI(), 'result')
    uv.CreateInput('scale', Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(2, 2))
    uv.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    for name in ['normal', 'roughness']:
        tex = UsdShade.Shader(stage.GetPrimAtPath(grain + '/' + name))
        tex.GetInput('st').ConnectToSource(uv.ConnectableAPI(), 'result')
        tex.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
    normal = stage.GetPrimAtPath(grain + '/normal')
    normal.GetAttribute('inputs:scale').Set(Gf.Vec4f(.8,.8,2,1))
    normal.GetAttribute('inputs:bias').Set(Gf.Vec4f(-.4,-.4,-1,0))


def apply(stage):
    from mug_rim_profile import apply as apply_rim
    root = stage.GetPrimAtPath(ROOT)
    apply_inner_glaze(stage)
    apply_outer_glaze(stage)
    if root.GetCustomDataByKey('mug17Profile') == REVISION:
        return {'revision': REVISION, 'height_unchanged': True, 'rim': apply_rim(stage)}
    previous = root.GetCustomDataByKey('mug17Profile')
    if previous not in (None, 'real_cup_gentle_curve_v2'):
        raise ValueError('Restore the original fixture before applying a new mug profile')
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        handle = 'Handle' in prim.GetName()
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float).copy()
            if handle:
                if previous is None:
                    points[:, 0] -= HANDLE_SHIFT
            elif '/Visuals/' in path or '/Colliders/Wall_' in path:
                radius = np.linalg.norm(points[:, :2], axis=1)
                delta = np.interp(points[:, 2], Z, DELTA)
                if previous:
                    delta -= np.interp(points[:, 2], Z, PREVIOUS_DELTA)
                # The inner floor contains a near-zero-radius pole ring.
                # Fade radial displacement there; adding a constant opens a hole.
                delta *= np.minimum(radius / .029, 1.)
                points[:, :2] *= (1 + delta / np.maximum(radius, 1e-12))[:, None]
            else:
                continue
            mesh.GetPointsAttr().Set(points.tolist())
            mesh.GetExtentAttr().Set([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
            # Recompute authored smooth normals, including duplicated UV seams.
            if mesh.GetNormalsAttr().HasAuthoredValueOpinion():
                counts = mesh.GetFaceVertexCountsAttr().Get()
                indices = mesh.GetFaceVertexIndicesAttr().Get()
                normals = np.zeros_like(points)
                offset = 0
                for count in counts:
                    face = list(indices[offset:offset+count]); offset += count
                    for j in range(1, count-1):
                        tri = [face[0], face[j], face[j+1]]
                        normal = np.cross(points[tri[1]]-points[tri[0]], points[tri[2]]-points[tri[0]])
                        normals[tri] += normal
                _, groups = np.unique(np.round(points, 8), axis=0, return_inverse=True)
                sums = np.zeros((groups.max()+1, 3)); np.add.at(sums, groups, normals)
                normals = sums[groups]
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
                mesh.GetNormalsAttr().Set(normals.tolist())
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        elif handle and prim.IsA(UsdGeom.Xformable) and previous is None:
            xf = UsdGeom.Xformable(prim)
            matrix = xf.GetLocalTransformation()
            translation = matrix.ExtractTranslation(); translation[0] -= HANDLE_SHIFT
            matrix.SetTranslateOnly(translation)
            xf.MakeMatrixXform().Set(matrix)
    root.SetCustomDataByKey('mug17Profile', REVISION)
    rim = apply_rim(stage)
    return {'revision': REVISION, 'height_unchanged': True,
            'reference': 'Three user close-ups; estimated belly/mouth ratio 1.118, not metrology',
            'mesh_height_m': .09, 'scene_height_m': .085,
            'mesh_mouth_outer_diameter_m': .088, 'mesh_belly_diameter_m': .0984,
            'source_modified': False, 'rim': rim}
