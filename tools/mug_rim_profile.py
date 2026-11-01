"""Thick ceramic rim and narrow black outer boundary from real-cup closeups.

Profile coordinates are millimetres before the existing 85/90 scene scale.
The neck thickens inward over the upper 18 mm, keeping overall height fixed.
"""
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdPhysics

ROOT = '/World/Household/Fixture'
REVISION = 'high_outer_low_inner_rim_v4'
NECK_REVISION = 'thick_ceramic_rim_v2'
HEIGHT_REVISION = 'lower_inner_edge_85mm_v1'
PROFILE_MM = np.array([
    [44, 89], [43.99, 89.35], [43.94, 89.7], [43.82, 89.92],
    [43.65, 90], [43.3, 89.9], [42.7, 89.55], [41.8, 88.8],
    [40.8, 87.85], [39.8, 86.95], [38.8, 86.2], [38, 85.7],
    [37.6, 85.4], [37.45, 85.2], [37.5, 85],
])


def update_mesh(mesh, points, smooth=False):
    mesh.GetPointsAttr().Set(points.tolist())
    mesh.GetExtentAttr().Set([Gf.Vec3f(*points.min(0)),Gf.Vec3f(*points.max(0))])
    if smooth:
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        face_normals = np.cross(points[indices[:,1]]-points[indices[:,0]],
                                points[indices[:,2]]-points[indices[:,0]])
        normals = np.zeros_like(points)
        for j in range(3): np.add.at(normals,indices[:,j],face_normals)
        _,groups = np.unique(np.round(points,8),axis=0,return_inverse=True)
        sums = np.zeros((groups.max()+1,3));np.add.at(sums,groups,normals)
        normals = sums[groups];normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        mesh.GetNormalsAttr().Set(normals.tolist())


def lower_inner_edge(stage):
    root = stage.GetPrimAtPath(ROOT)
    if root.GetCustomDataByKey('mugInnerEdgeHeight') == HEIGHT_REVISION:
        return
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        inside = path.endswith('/Visuals/Inside')
        wall = '/Colliders/Wall_' in path
        if not prim.IsA(UsdGeom.Mesh) or not (inside or wall):
            continue
        mesh = UsdGeom.Mesh(prim);points = np.array(mesh.GetPointsAttr().Get(),dtype=float)
        old_z = points[:,2].copy()
        if inside:
            points[:,2] = np.interp(old_z,[0,.072,.086,.089],[0,.072,.082,.085])
        else:
            radius = np.linalg.norm(points[:,:2],axis=1)
            for z in np.unique(old_z):
                at = old_z == z
                outside = at & (radius > (radius[at].min()+radius[at].max())/2)
                if z > .085:
                    new_z = .082 if z < .088 else .085
                    # Preserve the exterior body profile where the lower wall ends.
                    outer_radius = np.interp(new_z,[.072,.086],[.0469,.0446])
                    points[outside,:2] *= (outer_radius/radius[outside])[:,None]
                    points[at,2] = new_z
        update_mesh(mesh,points,smooth=inside)
    root.SetCustomDataByKey('mugInnerEdgeHeight',HEIGHT_REVISION)


def rim_collisions(stage):
    """Sloped lip wedges above the inner cavity; never a solid cap over its hole."""
    from scipy.spatial import ConvexHull
    # The convex wedge approximates only the small rounded profile, while its
    # sloping upper surface avoids the old invisible vertical collar above 85 mm.
    section = np.vstack([PROFILE_MM, [44.7642857,85]]) / 1000
    for i in range(64):
        angles = np.array([i,i+1])*(2*np.pi/64)
        points = np.array([[r*np.cos(a),r*np.sin(a),z] for a in angles for r,z in section])
        hull = ConvexHull(points)
        mesh = UsdGeom.Mesh.Define(stage,ROOT+'/Colliders/Rim_'+str(i))
        mesh.CreatePointsAttr(points.tolist());mesh.CreateFaceVertexCountsAttr([3]*len(hull.simplices))
        faces = hull.simplices.copy()
        for j,tri in enumerate(faces):
            n=np.cross(points[tri[1]]-points[tri[0]],points[tri[2]]-points[tri[0]])
            if np.dot(n,hull.equations[j,:3])<0: faces[j]=tri[::-1]
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateVisibilityAttr('invisible')
        mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)),Gf.Vec3f(*points.max(0))])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')


def thicken_neck(stage):
    root = stage.GetPrimAtPath(ROOT)
    if root.GetCustomDataByKey('mugNeckThickness') == NECK_REVISION:
        return
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or not (path.endswith('/Visuals/Inside') or '/Colliders/Wall_' in path):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        radius = np.linalg.norm(points[:,:2],axis=1)
        mask = np.ones(len(points), dtype=bool)
        if '/Colliders/' in path:
            # Keep exterior vertices fixed; thicken actual ceramic toward the cavity.
            for z in np.unique(points[:,2]):
                at = points[:,2] == z
                mask[at] = radius[at] < (radius[at].min()+radius[at].max())/2
        delta = np.interp(points[:,2],[0,.072,.086,.089,.09],[0,0,-.0027,-.004,-.004])
        points[mask,:2] *= (1+delta[mask]/np.maximum(radius[mask],1e-12))[:,None]
        mesh.GetPointsAttr().Set(points.tolist())
        mesh.GetExtentAttr().Set([Gf.Vec3f(*points.min(0)),Gf.Vec3f(*points.max(0))])
        if path.endswith('/Visuals/Inside'):
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
            face_normals = np.cross(points[indices[:,1]]-points[indices[:,0]],
                                    points[indices[:,2]]-points[indices[:,0]])
            normals = np.zeros_like(points)
            for j in range(3): np.add.at(normals,indices[:,j],face_normals)
            _,groups = np.unique(np.round(points,8),axis=0,return_inverse=True)
            sums = np.zeros((groups.max()+1,3));np.add.at(sums,groups,normals)
            normals = sums[groups];normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
            mesh.GetNormalsAttr().Set(normals.tolist())
    root.SetCustomDataByKey('mugNeckThickness',NECK_REVISION)


def apply(stage):
    thicken_neck(stage)
    lower_inner_edge(stage)
    rim_collisions(stage)
    path = ROOT + '/Visuals/WhiteLip'
    # Replace the old flat-topped white annulus, including stale UV attributes.
    stage.RemovePrim(path)
    mesh = UsdGeom.Mesh.Define(stage, path)
    control = PROFILE_MM / 1000
    arc = np.r_[0,np.cumsum(np.linalg.norm(np.diff(control,axis=0),axis=1))]
    derivatives = PchipInterpolator(arc, control, axis=0).derivative()(arc)
    # Match the authored wall normals at both joins instead of making a crease.
    for i, name in [(0,'OuterBody'),(-1,'Inside')]:
        wall = UsdGeom.Mesh(stage.GetPrimAtPath(ROOT+'/Visuals/'+name))
        wp = np.array(wall.GetPointsAttr().Get())
        wn = np.array(wall.GetNormalsAttr().Get())
        index = np.argmin(np.linalg.norm(wp-np.array([control[i,0],0,control[i,1]]),axis=1))
        derivatives[i] = [-wn[index,2],wn[index,0]]
    spline = CubicHermiteSpline(arc,control,derivatives,axis=0)
    samples_per_segment = 8
    samples = np.r_[np.concatenate([np.linspace(a,b,samples_per_segment,endpoint=False)
                                    for a,b in zip(arc[:-1],arc[1:])]),arc[-1]]
    profile = spline(samples)
    sides = 192
    angle = np.arange(sides) * (2 * np.pi / sides)
    points = np.array([[r*np.cos(a), r*np.sin(a), z]
                       for r, z in profile for a in angle])
    faces = []
    for j in range(len(profile)-1):
        for i in range(sides):
            k = (i+1) % sides
            a,b,c,d = j*sides+i,j*sides+k,(j+1)*sides+k,(j+1)*sides+i
            faces.extend([[a,b,c],[a,c,d]])
    tangent = spline.derivative()(samples)
    normals = np.array([[dz*np.cos(a),dz*np.sin(a),-dr]
                        for dr,dz in tangent for a in angle])
    normals /= np.linalg.norm(normals,axis=1)[:,None]
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.array(faces).ravel().tolist())
    mesh.CreateNormalsAttr(normals.tolist())
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)),Gf.Vec3f(*points.max(0))])
    material = UsdShade.Material.Define(stage,ROOT+'/Looks/RimBlackGlaze')
    shader = UsdShade.Shader.Define(stage,str(material.GetPath())+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.018,.0185,.019))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.26)
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    white = UsdShade.Material.Define(stage,ROOT+'/Looks/SmoothRimGlaze')
    glaze = UsdShade.Shader.Define(stage,str(white.GetPath())+'/Surface')
    glaze.CreateIdAttr('UsdPreviewSurface')
    color = stage.GetPrimAtPath(ROOT+'/Looks/WhiteGlaze/Surface').GetAttribute('inputs:diffuseColor').Get()
    glaze.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(color)
    glaze.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.14)
    glaze.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    white.CreateSurfaceOutput().ConnectToSource(glaze.ConnectableAPI(),'surface')
    binding.Bind(white)
    # Black ends at the crown: 0.35 mm radial width, white turns inward.
    boundary = 4 * samples_per_segment * sides * 2
    for name,indices,mat in [('OuterBlackEdge',range(boundary),material),
                             ('InnerWhiteGlaze',range(boundary,len(faces)),white)]:
        subset = binding.CreateMaterialBindSubset(name,list(indices))
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mat)
    binding.SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.partition)
    stage.GetPrimAtPath(ROOT).SetCustomDataByKey('mugRimProfile',REVISION)
    return {'revision':REVISION,'mesh_outer_black_width_m':.00035,
            'mesh_radial_wall_thickness_m':.0065,'scene_radial_wall_thickness_m':.0065*85/90,
            'scene_inner_opening_diameter_m':.075*85/90,
            'mesh_max_height_m':.09,'scene_height_m':.085,
            'scene_outer_inner_height_difference_m':.005*85/90,
            'reference':'High outer edge slopes smoothly to a lower inner edge; dimensions estimated from reference photos',
            'rim_roughness':.14,'collision_changed':True}
