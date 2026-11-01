"""Low-complexity free-body contacts for the L1-03 visor and supporting box.

Visual meshes, hat mass and central head opening are preserved. The box mass
is an explicit simulation tuning value, not a measured real-world mass.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = '/World/Household'
REVISION = 'five_box_support_open_visor_v1'
BOX_MASS = .6
# Full sizes and centres, metres. Shallow recess retained above the body.
BOXES = [
    ('Body', (.1996,.19012,.070125), (0,0,.0350625)),
    ('EdgeLeft', (.006,.195,.004875), (-.0995,0,.0725625)),
    ('EdgeRight', (.006,.195,.004875), (.0995,0,.0725625)),
    ('EdgeFront', (.193,.006,.004875), (0,-.0945,.0725625)),
    ('EdgeBack', (.193,.006,.004875), (0,.0945,.0725625)),
]


def contact(prim, material):
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    prim.AddAppliedSchema('PhysxCollisionAPI')
    prim.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.0003)
    prim.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
    UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')


def hull_mesh(stage,path,points,material):
    points = np.unique(points,axis=0)
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    for i,tri in enumerate(faces):
        normal = np.cross(vertices[tri[1]]-vertices[tri[0]],vertices[tri[2]]-vertices[tri[0]])
        if np.dot(normal,hull.equations[i,:3])<0: faces[i]=tri[::-1]
    mesh = UsdGeom.Mesh.Define(stage,path)
    mesh.CreatePointsAttr(vertices.tolist());mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(*vertices.min(0)),Gf.Vec3f(*vertices.max(0))])
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    contact(mesh.GetPrim(),material)
    return vertices


def apply(stage):
    native = Usd.Stage.Open(str(Path(__file__).resolve().parents[1]/'assets/tuojing/source/L1-07/scene.usdz'))
    for name in ['source_box','hat']:
        stage.RemovePrim(f'{ROOT}/{name}/Colliders')
        UsdGeom.Xform.Define(stage,f'{ROOT}/{name}/Colliders')
        mat = UsdShade.Material(stage.GetPrimAtPath(f'{ROOT}/{name}/Looks/Contact'))
        UsdPhysics.MaterialAPI.Apply(mat.GetPrim()).CreateRestitutionAttr(0.)
        body = stage.GetPrimAtPath(f'{ROOT}/{name}')
        body.AddAppliedSchema('PhysxRigidBodyAPI')
        for key,val,kind in [('maxDepenetrationVelocity',.3,Sdf.ValueTypeNames.Float),
                             ('solverPositionIterationCount',16,Sdf.ValueTypeNames.Int),
                             ('solverVelocityIterationCount',4,Sdf.ValueTypeNames.Int),
                             ('enableCCD',True,Sdf.ValueTypeNames.Bool)]:
            body.CreateAttribute('physxRigidBody:'+key,kind).Set(val)
    boxmat = UsdShade.Material(stage.GetPrimAtPath(ROOT+'/source_box/Looks/Contact'))
    for name,size,centre in BOXES:
        cube = UsdGeom.Cube.Define(stage,ROOT+'/source_box/Colliders/'+name)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre));cube.AddScaleOp().Set(Gf.Vec3f(*size))
        contact(cube.GetPrim(),boxmat)
    sizes=np.array([x[1] for x in BOXES]);centres=np.array([x[2] for x in BOXES])
    masses=np.prod(sizes,axis=1);masses*=BOX_MASS/masses.sum()
    com=(centres*masses[:,None]).sum(0)/BOX_MASS;offset=centres-com
    inertia=(masses[:,None]*((np.sum(sizes*sizes,axis=1)[:,None]-sizes*sizes)/12+
                            np.sum(offset*offset,axis=1)[:,None]-offset*offset)).sum(0)
    mass=UsdPhysics.MassAPI(stage.GetPrimAtPath(ROOT+'/source_box'))
    mass.CreateMassAttr(BOX_MASS);mass.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia));mass.CreatePrincipalAxesAttr(Gf.Quatf(1.))
    hatmat=UsdShade.Material(stage.GetPrimAtPath(ROOT+'/hat/Looks/Contact'))
    source={'_'.join([p.GetName().split('_')[0]]+[str(int(x)) for x in p.GetName().split('_')[1:]]):np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=float)
            for p in Usd.PrimRange(native.GetPrimAtPath(ROOT+'/hat/Colliders')) if p.IsA(UsdGeom.Mesh)}
    hat_parts=[]
    for index in range(32):
        parts=[source['Band_'+str(i)] for i in range(index*2,index*2+2)]
        hat_parts.append(hull_mesh(stage,f'{ROOT}/hat/Colliders/Band_{index}',np.vstack(parts),hatmat))
    for index in range(12):
        parts=[source[f'Brim_{i}_{j}'] for i in range(index*4,index*4+4) for j in range(3)]
        hat_parts.append(hull_mesh(stage,f'{ROOT}/hat/Colliders/Brim_{index}',np.vstack(parts),hatmat))
    # Check the initial support in box coordinates using convex projection, not
    # an AABB that would fill the head opening. Raise only for actual overlap.
    cache=UsdGeom.XformCache();hatprim=stage.GetPrimAtPath(ROOT+'/hat')
    hatworld=np.array(cache.GetLocalToWorldTransform(hatprim)).T
    boxworld=np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT+'/source_box'))).T
    relative=np.linalg.inv(boxworld)@hatworld
    lift=0.
    for points in hat_parts:
        vertices=points@relative[:3,:3].T+relative[:3,3]
        for _,size,centre in BOXES:
            lo=np.array(centre)-np.array(size)/2;hi=np.array(centre)+np.array(size)/2
            if np.any(vertices[:,:2].max(0)<lo[:2]) or np.any(vertices[:,:2].min(0)>hi[:2]):continue
            h=ConvexHull(vertices)
            a=np.vstack([h.equations[:,:3],[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]])
            b=np.r_[-h.equations[:,3],hi[0],-lo[0],hi[1],-lo[1]]
            fit=linprog([0,0,1],A_ub=a,b_ub=b,bounds=[(None,None)]*3,method='highs')
            if fit.success:lift=max(lift,hi[2]+.0004-fit.x[2])
    if lift>0:
        if lift>.015:raise ValueError(f'Unexpected initial overlap requires {lift:.4f} m lift')
        hatworld[:3,3]+=boxworld[:3,2]*lift
        parent=np.array(cache.GetLocalToWorldTransform(hatprim.GetParent())).T
        UsdGeom.Xformable(hatprim).MakeMatrixXform().Set(Gf.Matrix4d((np.linalg.inv(parent)@hatworld).T.tolist()))
    return {'revision':REVISION,'box_colliders':5,'box_mass_kg':BOX_MASS,
            'box_mass_basis':'simulation stability tuning; not a measured mass',
            'hat_colliders':44,'hat_band_sectors':32,'hat_brim_sectors':12,
            'hat_mass_kg':.065,'head_opening_preserved':True,'visual_meshes_changed':False,
            'initial_hat_clearance_lift_m':lift,'contact_offset_m':.0003,
            'restitution':0.,'max_depenetration_velocity_m_s':.3,
            'free_dynamic_bodies':True}
