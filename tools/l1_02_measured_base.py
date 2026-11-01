"""User-measured moulded plug holder. Dimensions in metres; photo details estimated."""
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Gf, Usd, UsdGeom, UsdShade, UsdPhysics

ROOT = '/World/Household/plug_storage_base'
REVISION = 'measured_83_93_55_cavity45_depth25_20261016_v1'
N = 128
THETA = np.arange(N) * 2 * np.pi / N
DIRECTIONS = np.column_stack((np.cos(THETA), np.sin(THETA)))


def rounded_sdf(p, half, radius):
    q = np.abs(p) - (np.asarray(half) - radius)
    return np.linalg.norm(np.maximum(q, 0.), axis=-1) + np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.) - radius


def radial(sdf):
    low, high = np.zeros(N), np.full(N, .08)
    for _ in range(36):
        mid = (low + high) / 2
        inside = sdf(DIRECTIONS * mid[:, None]) <= 0
        low = np.where(inside, mid, low)
        high = np.where(inside, high, mid)
    return (low + high) / 2


def rr(half, radius):
    return radial(lambda p: rounded_sdf(p, [half, half], radius))


def opening_sdf(p):
    # 45 mm main square; two estimated 20 mm diameter finger scallops,
    # and a small diagonal moulded notch at the front-right corner.
    square = rounded_sdf(p, [.0225, .0225], .004)
    lobes = [np.linalg.norm(p-[0., sign*.021], axis=-1)-.010 for sign in [-1, 1]]
    angle = np.deg2rad(-45)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    notch = rounded_sdf((p-[.0215, -.0215]) @ rot, [.006, .008], .002)
    return np.minimum.reduce([square, *lobes, notch])


def rings():
    floor = rr(.0225, .004)
    opening = radial(opening_sdf)
    middle = floor + (opening-floor)*.50
    # Outer surface from foot to flange, down the moulded well to the hole.
    outer = [(rr(.0465,.004),0.), (rr(.0465,.004),.001),
             (rr(.0415,.005),.0535), (rr(.0405,.0045),.055),
             (opening+.0015,.055), (opening,.0535),
             (middle,.034), (floor,.030), (np.full(N,.010),.030)]
    # Back of a 1.5 mm moulded shell; centre hole opens into hollow underside.
    inner = [(rr(.045,.003),0.), (rr(.045,.003),.001),
             (rr(.040,.0035),.052), (rr(.0395,.0035),.0535),
             (opening+.003,.0535), (opening+.0015,.052),
             (middle+.0015,.033), (floor+.0007,.0285), (np.full(N,.010),.0285)]
    def points(rows):
        return [np.column_stack((DIRECTIONS*r[:,None],np.full(N,z))) for r,z in rows]
    return points(outer), points(inner)


def mesh(stage, path, points, faces, smooth=False):
    points = np.asarray(points, dtype=float)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(points.tolist())
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    m.CreateSubdivisionSchemeAttr('none')
    m.CreateExtentAttr([Gf.Vec3f(*points.min(0)),Gf.Vec3f(*points.max(0))])
    if smooth:
        normals = np.zeros_like(points)
        for f in faces:
            normal = np.cross(points[f[1]]-points[f[0]],points[f[2]]-points[f[0]])
            normals[f] += normal
        normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        m.CreateNormalsAttr(normals.tolist());m.SetNormalsInterpolation('vertex')
    return m


def apply(stage):
    root = stage.GetPrimAtPath(ROOT)
    if not root:
        raise ValueError('Missing L1-02 holder')
    for name in ['Visuals', 'Colliders']:
        stage.RemovePrim(ROOT+'/'+name)
        UsdGeom.Xform.Define(stage, ROOT+'/'+name)
    material = UsdShade.Material.Define(stage,ROOT+'/Looks/MeasuredBlackPlastic')
    shader = UsdShade.Shader.Define(stage,str(material.GetPath())+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    from pxr import Sdf
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.014,.014,.014))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.33)
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.48)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    upper, lower = rings()
    rows = upper+list(reversed(lower))
    points = np.concatenate(rows);faces=[]
    for k in range(len(rows)):
        k1=(k+1)%len(rows)
        for j in range(N):
            j1=(j+1)%N
            faces.append([k*N+j,k*N+j1,k1*N+j1,k1*N+j])
    visual=mesh(stage,ROOT+'/Visuals/MouldedShell',points,faces,True)
    # Keep the broad flange and pocket floor optically flat; interpolating
    # their normals with tall neighbouring walls creates false dents/highlights.
    normals = np.asarray(visual.GetNormalsAttr().Get())
    corner_normals = []
    for face_index, face in enumerate(faces):
        band = face_index // N
        if band in (3, 7):
            corner_normals.extend([[0., 0., 1.]] * 4)
        elif band in (9, 13):
            corner_normals.extend([[0., 0., -1.]] * 4)
        elif band == 8:
            for vertex in face:
                xy=points[vertex,:2];xy=-xy/np.linalg.norm(xy)
                corner_normals.append([*xy,0.])
        else:
            corner_normals.extend(normals[face].tolist())
    visual.GetNormalsAttr().Set(corner_normals)
    visual.SetNormalsInterpolation('faceVarying')
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
    contact=UsdShade.Material(stage.GetPrimAtPath(ROOT+'/Looks/Contact'))
    # Each cell is a convex shell segment. No hull spans the central cavity or
    # through-hole; the visible well and the physical support share the rings.
    count=0
    for k in range(len(upper)-1):
        for j in range(N):
            j1=(j+1)%N
            v=np.array([upper[k][j],upper[k][j1],upper[k+1][j],upper[k+1][j1],
                        lower[k][j],lower[k][j1],lower[k+1][j],lower[k+1][j1]])
            hull=ConvexHull(v)
            triangles=[]
            for tri, eq in zip(hull.simplices,hull.equations):
                tri=tri.tolist()
                if np.dot(np.cross(v[tri[1]]-v[tri[0]],v[tri[2]]-v[tri[0]]),eq[:3])<0:tri.reverse()
                triangles.append(tri)
            collider=mesh(stage,ROOT+f'/Colliders/Shell_{k:02d}_{j:03d}',v,triangles)
            collider.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr('convexHull')
            collider.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
            collider.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.0003)
            collider.GetPrim().CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
            UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(contact,materialPurpose='physics')
            count+=1
    for name,z in [('PlugSeat',.030),('ThroughHole',.030),('Origin',0.)]:
        frame=UsdGeom.Xform.Define(stage,ROOT+'/Frames/'+name)
        matrix=Gf.Matrix4d(1);matrix.SetTranslate(Gf.Vec3d(0,0,z));frame.MakeMatrixXform().Set(matrix)
    # Pins enter the central hole; housing rests on the 30 mm-high well floor.
    # This is initial placement only, never a pose write during simulation.
    cache=UsdGeom.XformCache();base=np.asarray(cache.GetLocalToWorldTransform(root)).T
    plug=stage.GetPrimAtPath('/World/Household/plug')
    if plug:
        world=np.asarray(cache.GetLocalToWorldTransform(plug)).T
        world[:3,3]=base[:3,3]+[0,0,.0135]
        parent=np.asarray(cache.GetLocalToWorldTransform(plug.GetParent())).T
        UsdGeom.Xformable(plug).MakeMatrixXform().Set(Gf.Matrix4d((np.linalg.inv(parent)@world).T.tolist()))
    root.SetCustomDataByKey('measuredHolderRevision',REVISION)
    return {'revision':REVISION,'measured':{'top_xy_m':[.083,.083], 'bottom_xy_m':[.093,.093],
        'height_m':.055,'central_well_xy_m':[.045,.045],'well_depth_m':.025},
        'photo_estimates':{'hole_diameter_m':.020,'scallop_radius_m':.010,'shell_thickness_m':.0015,
                          'notch':'rounded diagonal front-right relief','outer_corner_radius_m':.005},
        'collision_convex_cells':count,'mass_and_contact_material_preserved':True,
        'source_modified':False,'plug_housing_clearance_above_floor_m':.0005}
