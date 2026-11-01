"""Scene-only support inferred from the real charger exposure, not a measured part."""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
from l1_02_measured_base import ROOT, mesh

HEIGHT = .015


def apply(stage):
    # A ring attached to the holder supports the housing while leaving both
    # metal blades free. It is a compound collision shape, never an attachment
    # to the charger. The measured moulded holder itself remains unchanged.
    material = UsdShade.Material(stage.GetPrimAtPath(ROOT+'/Looks/MeasuredBlackPlastic'))
    contact = UsdShade.Material(stage.GetPrimAtPath(ROOT+'/Looks/Contact'))
    count = 32
    faces = [[0,3,2,1], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]]
    all_points, all_faces = [], []
    for j in range(count):
        angles = np.array([j,j+1]) * 2*np.pi/count
        xy = np.array([[r*np.cos(a),r*np.sin(a)] for r,a in
                       [(.010,angles[0]),(.021,angles[0]),(.021,angles[1]),(.010,angles[1])]])
        points = np.concatenate([np.column_stack((xy,np.full(4,z))) for z in (.030,.030+HEIGHT)])
        offset = len(all_points)
        all_points.extend(points.tolist()); all_faces.extend([[offset+i for i in f] for f in faces])
        collider = mesh(stage,ROOT+f'/Colliders/HeightSpacer_{j:02d}',points,faces)
        collider.CreateVisibilityAttr('invisible')
        prim = collider.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
        prim.AddAppliedSchema('PhysxCollisionAPI')
        prim.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.0003)
        prim.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(contact,materialPurpose='physics')
    visual = mesh(stage,ROOT+'/Visuals/HeightSpacer',all_points,all_faces)
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
    cache = UsdGeom.XformCache()
    base = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))).T
    plug = stage.GetPrimAtPath('/World/Household/plug')
    world = np.asarray(cache.GetLocalToWorldTransform(plug)).T
    world[:3,3] = base[:3,3]+[0,0,.0135+HEIGHT]
    parent = np.asarray(cache.GetLocalToWorldTransform(plug.GetParent())).T
    UsdGeom.Xformable(plug).MakeMatrixXform().Set(Gf.Matrix4d((np.linalg.inv(parent)@world).T.tolist()))
    frame = UsdGeom.Xform.Define(stage,ROOT+'/Frames/PlugSeat')
    matrix = Gf.Matrix4d(1); matrix.SetTranslate(Gf.Vec3d(0,0,.030+HEIGHT))
    frame.MakeMatrixXform().Set(matrix)
    return {'revision':'photo_estimated_support_15mm_20261016_v1',
            'height_m':HEIGHT,'outer_diameter_m':.042,'pin_channel_diameter_m':.020,
            'support_top_above_holder_bottom_m':.045,
            'provenance':'15 mm height estimated from photo; concealed real support shape/material unconfirmed',
            'physics':'Holder-attached compound ring; charger remains an independent dynamic body',
            'holder_measured_dimensions_unchanged':True}
