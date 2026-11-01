"""Third-trial brush grip proxy: open annular cylinder plus flat neck box."""
import math

def apply(stage):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
    root = stage.GetPrimAtPath('/World/L2/broom')
    if not root:
        raise ValueError('Missing brush')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
    disabled = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/L2/broom/Colliders')):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not prim.GetName().startswith(('MoldedWeb', 'OuterRaisedRail')):
            continue
        box = cache.ComputeRelativeBound(prim, root).ComputeAlignedBox()
        if box.GetMin()[1] < -.060:
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            disabled.append(str(prim.GetPath()))
    if not disabled:
        raise ValueError('Expected detailed grip colliders')
    parent = '/World/L2/broom/HandleProxy'
    material = UsdShade.Material(stage.GetPrimAtPath('/World/L2/broom/Looks/Contact'))
    created = []
    def collision(prim):
        UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(.0002)
        api.CreateRestOffsetAttr(0.)
        if material:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
        created.append(str(prim.GetPath()))
    # A single convex hull would incorrectly fill the hanging hole. Each sector
    # is instead one convex prism, forming a compound, truly hollow cylinder.
    inner, outer, cy, height, count = .0095, .0135, -.0905, .006, 16
    for i in range(count):
        angles = [2*math.pi*i/count, 2*math.pi*(i+1)/count]
        xy = [(r*math.cos(a),cy+r*math.sin(a)) for r,a in
              [(inner,angles[0]),(outer,angles[0]),(outer,angles[1]),(inner,angles[1])]]
        points = [(x,y,z) for z in [0.,height] for x,y in xy]
        mesh = UsdGeom.Mesh.Define(stage, parent+'/Ring%02d'%i)
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        mesh.CreateFaceVertexCountsAttr([4]*6)
        mesh.CreateFaceVertexIndicesAttr([3,2,1,0,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
        mesh.CreateSubdivisionSchemeAttr('none')
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        collision(mesh.GetPrim())
    cube = UsdGeom.Cube.Define(stage, parent+'/FlatNeck')
    cube.CreateSizeAttr(1.)
    xf = UsdGeom.Xformable(cube);xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.,(-.0805-.057)/2,height/2))
    xf.AddScaleOp().Set(Gf.Vec3f(.018,.0235,height))
    collision(cube.GetPrim())
    return {'protocol':'hollow_flat_ring_and_box_v1','disabled_count':len(disabled),
            'disabled':disabled,'created':created,'ring_outer_radius_m':outer,
            'ring_inner_radius_m':inner,'thickness_m':height,
            'neck_size_m':[.018,.0235,height], 'visuals_modified':False,
            'brush_head_and_bristles_unchanged':True,'mass_and_drive_parameters_unchanged':True}
