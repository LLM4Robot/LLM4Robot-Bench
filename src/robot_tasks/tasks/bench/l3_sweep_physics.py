"""L3-04 support-box collision only: body plus four raised rim boxes."""
import numpy as np


def apply(stage):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade

    root_path = '/World/L2/support_box'
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError('Missing L3-04 support box')
    source = root_path + '/Colliders'
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'proxy', 'render'])

    def bounds(path):
        p = stage.GetPrimAtPath(path)
        if not p:
            raise ValueError('Missing source collider: ' + path)
        b = cache.ComputeRelativeBound(p, root).ComputeAlignedBox()
        return np.array(b.GetMin()), np.array(b.GetMax())

    body_lo, body_hi = bounds(source + '/LowerBody')
    inset_lo, inset_hi = bounds(source + '/InsetFace')
    rim_bounds = [bounds(str(p.GetPath())) for p in stage.GetPrimAtPath(source).GetChildren()
                  if p.GetName().startswith('Rim_')]
    if not rim_bounds:
        raise ValueError('Missing source rim geometry')
    outer_lo = np.min([lo for lo, hi in rim_bounds], axis=0)
    outer_hi = np.max([hi for lo, hi in rim_bounds], axis=0)
    floor_z, top_z = float(inset_hi[2]), float(outer_hi[2])
    if not (body_lo[2] < floor_z < top_z and
            np.all(outer_lo[:2] < inset_lo[:2]) and
            np.all(inset_hi[:2] < outer_hi[:2])):
        raise ValueError('Unexpected L3-04 box axes/geometry')
    # Preserve the central recessed top. Do not turn the whole raised rim into
    # one solid lid at the rim's highest elevation.
    body_hi[2] = floor_z
    boxes = [('Body', body_lo, body_hi),
        ('RimLeft', [outer_lo[0], outer_lo[1], floor_z], [inset_lo[0], outer_hi[1], top_z]),
        ('RimRight', [inset_hi[0], outer_lo[1], floor_z], [outer_hi[0], outer_hi[1], top_z]),
        ('RimNear', [inset_lo[0], outer_lo[1], floor_z], [inset_hi[0], inset_lo[1], top_z]),
        ('RimFar', [inset_lo[0], inset_hi[1], floor_z], [inset_hi[0], outer_hi[1], top_z])]
    disabled = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(source)):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
            disabled.append(str(p.GetPath()))
    material = UsdShade.Material(stage.GetPrimAtPath(root_path + '/Looks/Contact'))
    result = []
    for name, lo, hi in boxes:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        cube = UsdGeom.Cube.Define(stage, root_path + '/BoxColliders/' + name)
        cube.CreateSizeAttr(1.)
        xf = UsdGeom.Xformable(cube)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*((lo+hi)/2)))
        xf.AddScaleOp().Set(Gf.Vec3f(*(hi-lo)))
        cube.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        api = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        api.CreateContactOffsetAttr(.0002)
        api.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose='physics')
        result.append({'path':str(cube.GetPath()), 'center_local_m':((lo+hi)/2).tolist(),
                       'size_m':(hi-lo).tolist()})
    return {'protocol':'l3_05_support_box_five_boxes_v1', 'disabled_original_count':len(disabled),
            'boxes':result, 'visuals_modified':False, 'body_kinematic_unchanged':True,
            'limitation':'Rectangular approximation of rounded/sloped edges; central recess preserved.'}
