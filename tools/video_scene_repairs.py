"""Repair duplicate physical bindings in the immutable October 15 delivery.

Applied once when building the runtime USD; source USDZ files are never edited.
All operations describe episode setup, never attach or move props during play.
"""
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

REVISION = 'latest-prop-bindings-fixed-bottle-board-v1'


def move_prim(stage, source, target):
    editor = Usd.NamespaceEditor(stage)
    editor.MovePrimAtPath(source, target)
    if not editor.CanApplyEdits():
        raise ValueError(f'Cannot rebind delivered geometry: {source} -> {target}')
    editor.ApplyEdits()


def matrix(stage, path):
    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T


def replace_support(stage, row):
    source = '/World/ReviewAssets/MeasuredBox'
    if not stage.GetPrimAtPath(source):
        return None
    target = next(v['path'] for v in row['fixtures'] if v['path'].endswith(('/support_box', '/Pedestal')))
    # Removing the old hidden support removes its colliders too. Move the new
    # measured box into the existing fixture binding, updating material paths.
    stage.RemovePrim(target)
    move_prim(stage, source, target)
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(target)).CreateKinematicEnabledAttr(True)
    stage.GetPrimAtPath(target).GetAttribute('physxRigidBody:enableCCD').Set(False)
    return {'fixture': target, 'source': source,
            'kinematic': True, 'duplicate_colliders_removed': True,
            'geometry_and_materials': 'latest complete delivery, unchanged'}


def split_marker(stage):
    source = '/World/ReviewAssets/MeasuredMarker'
    native = '/World/L2/marker'
    latest_world = matrix(stage, source)
    UsdGeom.Imageable(stage.GetPrimAtPath(native)).MakeVisible()
    move_prim(stage, source+'/Body/Looks', native+'/LatestLooks')
    copied = {'Body': [], 'Cap': []}
    for name in copied:
        root = native+'/'+name
        # Keep the functional nib/neck revealed on uncapping. Exterior visuals
        # all come from the new measured marker, with their exact world poses.
        for child in stage.GetPrimAtPath(root+'/Visuals').GetChildren():
            if not (name == 'Body' and child.GetName() in ('Nib', 'Neck')):
                UsdGeom.Imageable(child).MakeInvisible()
        parent = UsdGeom.Xform.Define(stage, root+'/LatestVisuals')
        local = np.linalg.inv(matrix(stage, root)) @ latest_world
        parent.MakeMatrixXform().Set(Gf.Matrix4d(local.T.tolist()))
    layer = stage.GetRootLayer()
    for prim in stage.GetPrimAtPath(source+'/Body/Visuals').GetChildren():
        body = 'Cap' if prim.GetName().startswith('Cap') else 'Body'
        target = native+'/'+body+'/LatestVisuals/'+prim.GetName()
        Sdf.CopySpec(layer, prim.GetPath(), layer, Sdf.Path(target))
        copied[body].append(prim.GetName())
    # The complete marker's barrel exterior can be used directly. Its solid
    # cap collider cannot: it would fill the insertion cavity. Retain the
    # supplied detachable cap's hollow fit and passive spring shoes instead.
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath(native+'/Body/Colliders/Barrel')).CreateCollisionEnabledAttr(False)
    parent = UsdGeom.Xform.Define(stage, native+'/Body/Colliders/LatestExterior')
    local = np.linalg.inv(matrix(stage,native+'/Body')) @ latest_world
    parent.MakeMatrixXform().Set(Gf.Matrix4d(local.T.tolist()))
    for name in ('Tail','Collar','Barrel'):
        Sdf.CopySpec(layer, Sdf.Path(source+'/Body/Colliders/'+name), layer,
                     Sdf.Path(str(parent.GetPath())+'/'+name))
    # The new barrel ends at 106.25 mm in the retained body frame, while the
    # original detachable cap began at 105 mm. That overlap wedges the cap
    # open and can launch both bodies. Fit the hollow cap axially to the new
    # visible shell and move its passive spring anchors by the same mapping.
    old_low,old_high,new_low,new_high = .105,.151,.10625,.14875
    scale=(new_high-new_low)/(old_high-old_low)
    offset=new_low-scale*old_low
    for prim in Usd.PrimRange(stage.GetPrimAtPath(native+'/Cap/Colliders')):
        if prim.IsA(UsdGeom.Mesh):
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            points[:,2]=points[:,2]*scale+offset
            UsdGeom.Mesh(prim).GetPointsAttr().Set(points.tolist())
    envelope=UsdGeom.Cylinder(stage.GetPrimAtPath(native+'/Cap/BoxContactEnvelope'))
    envelope.GetHeightAttr().Set(new_high-new_low)
    xf=UsdGeom.Xformable(envelope);transform=np.asarray(xf.GetLocalTransformation()).T
    transform[2,3]=(new_low+new_high)/2
    xf.MakeMatrixXform().Set(Gf.Matrix4d(transform.T.tolist()))
    for i in range(4):
        shoe=stage.GetPrimAtPath(native+'/Shoe'+str(i));xf=UsdGeom.Xformable(shoe)
        transform=np.asarray(xf.GetLocalTransformation()).T
        transform[2,3]=transform[2,3]*scale+offset
        xf.MakeMatrixXform().Set(Gf.Matrix4d(transform.T.tolist()))
        joint=UsdPhysics.Joint(stage.GetPrimAtPath(native+'/RadialSpring'+str(i)))
        anchor=list(joint.GetLocalPos0Attr().Get());anchor[2]=anchor[2]*scale+offset
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*anchor))
    for i in range(4):
        UsdGeom.Imageable(stage.GetPrimAtPath(native+'/Shoe'+str(i))).MakeInvisible()
    stage.RemovePrim(source)
    return {'latest_visual_parts': copied, 'one_piece_duplicate_removed': True,
            'body_and_cap': [native+'/Body',native+'/Cap'],
            'physical_fit': 'native detachable hollow cap and passive retention anchors fitted axially to the latest shell; exposed nib retained',
            'cap_axial_bounds_m': [new_low,new_high],
            'limitation': 'Outer visuals and barrel are the measured release; the supplied functional hollow cap collision approximates the measured outer shell.'}


def rotate_about_origin(stage, path, radians):
    prim = stage.GetPrimAtPath(path)
    transform = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T
    c,s = np.cos(radians),np.sin(radians)
    transform[:3,:3] = np.array([[c,-s,0],[s,c,0],[0,0,1.]]) @ transform[:3,:3]
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(transform.T.tolist()))


def align_l108_right_cup(stage):
    """Move the overview-right cup 10 mm rearward, once, at scene build time."""
    path = '/World/Household/lower_cup'
    prim = stage.GetPrimAtPath(path)
    world = matrix(stage, path)
    world[1, 3] = 1.1819986910835127 + .010
    parent = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(
        Gf.Matrix4d((np.linalg.inv(parent) @ world).T.tolist()))
    return {'revision': 'right_cup_rearward_10mm_20261016_v1',
            'object': 'lower_cup', 'view': 'overview right',
            'world_translation_delta_m': [0., .010, 0.],
            'geometry_materials_and_physics_unchanged': True}


def fix_l109_cups(stage, row):
    """Keep both flower cups at their authored table poses throughout an episode."""
    paths = []
    for name in ('source_cup', 'target_cup'):
        path = row['tracked'][name]['path']
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Missing L1-05 cup rigid body: ' + path)
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
        # PhysX does not support CCD on kinematic bodies; their collision
        # shapes still support the dynamic flower, charger and glass beads.
        prim.CreateAttribute('physxRigidBody:enableCCD', Sdf.ValueTypeNames.Bool).Set(False)
        paths.append(path)
    return {'revision': 'both_cups_fixed_to_table_20261018_v1',
            'paths': paths, 'kinematic': True,
            'poses_geometry_and_materials_unchanged': True,
            'flower_charger_and_beads_remain_dynamic': True}


def apply(stage, tid, row):
    result = {'revision': REVISION, 'source_modified': False}
    if tid == 'L1-01':
        from mug17_profile import apply as reshape_mug
        result['mug_profile'] = reshape_mug(stage)
        from yagu_marker_appearance import apply as marker_label
        row['appearance_overrides'] = {'yagu_marker': marker_label(stage)}
    box = replace_support(stage, row)
    if box:
        result['support'] = box
    if tid == 'L1-03':
        from l1_03_initial_layout import apply as align_hat_and_rack
        result['initial_layout'] = align_hat_and_rack(stage)
        from l1_03_contact_geometry import apply as simplify_hat_contacts
        result['contact_geometry'] = simplify_hat_contacts(stage)
    if tid == 'L1-02':
        from l1_02_measured_base import apply as measured_holder
        result['measured_holder'] = measured_holder(stage)
        from l1_02_height_spacer import apply as height_spacer
        result['height_spacer'] = height_spacer(stage)
    if tid == 'L1-04':
        result['initial_layout'] = align_l108_right_cup(stage)
    if tid == 'L1-05':
        result['fixed_cups'] = fix_l109_cups(stage, row)
    if tid == 'L2-01':
        result['marker'] = split_marker(stage)
        from l2_marker_asset import apply as replace_marker_exterior
        result['marker']['asset_override'] = replace_marker_exterior(stage)
    if tid == 'L2-02':
        # Contact reporting deliberately disables sleep. Resolve the contact
        # velocity more accurately instead of freezing or hiding jitter.
        for name in ('lower_book','middle_book','upper_book'):
            prim=stage.GetPrimAtPath(row['tracked'][name]['path'])
            prim.GetAttribute('physxRigidBody:solverPositionIterationCount').Set(64)
            prim.GetAttribute('physxRigidBody:solverVelocityIterationCount').Set(16)
        result['books']={'position_iterations': 64, 'velocity_iterations': 16,
                         'remain_dynamic': True, 'continuous_contact_reports': True,
                         'mass_friction_and_restitution_unchanged': True}
    if tid == 'L3-01':
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(row['tracked']['bottle']['path'])).CreateKinematicEnabledAttr(True)
        result['bottle_fixed'] = True
    if tid == 'L2-07':
        rotate_about_origin(stage,row['fixtures'][0]['path'],np.pi)
        rotate_about_origin(stage,row['tracked']['eraser']['path'],np.pi/2)
        result['video_orientation'] = {'board_yaw_delta_rad': float(np.pi), 'eraser_yaw_delta_rad': float(np.pi/2),
                                       'ink_and_scoring_follow_board_frame': True}
    from l2_measured_support import apply as measure_support
    measured = measure_support(stage, tid, row)
    if measured:
        row['l2_support_measurement'] = measured
        result['support']['geometry_and_materials'] = 'Latest complete delivery; L2 dimensions corrected to user measurement, materials preserved'
    from l2_video_placement import apply as place_from_video
    placement = place_from_video(stage, tid, row)
    if placement:
        row['video_placement_refinement'] = placement
    from video_scene_metadata import record
    row['tracked'] = {name: record(stage,v['path']) for name,v in row['tracked'].items()}
    row['fixtures'] = [record(stage,v['path']) for v in row['fixtures']]
    if measured and 'orange' in row:
        row['orange'] = record(stage, row['orange']['path'])
    return result
