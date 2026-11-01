"""Simplify the delivered L2-01 cap without filling its insertion cavity.

Only imported runtime layers are edited. Visuals, mass, inertia, friction,
barrel/neck geometry and scoring are unchanged. The passive radial shoes retain
their stiffness and travel; they model internal fit, not external grippable parts.
"""

import math
import numpy as np


def simplify_delivered_marker(stage, marker_path):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    cap_path = marker_path + '/Cap'
    root = stage.GetPrimAtPath(cap_path + '/Colliders')
    walls = [p for p in root.GetChildren() if p.GetName().startswith('Wall')]
    roof = [p for p in root.GetChildren()
            if p.GetName() == 'RoofCenter' or p.GetName().startswith('TopRing_')]
    if len(walls) != 16 or len(roof) != 225:
        raise ValueError('Unexpected delivered marker topology; audit before simplifying')
    wall_points = np.concatenate([np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get()) for p in walls])
    roof_points = np.concatenate([np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get()) for p in roof])
    radius = float(np.linalg.norm(wall_points[:, :2], axis=1).max())
    lower, upper = map(float, (roof_points[:, 2].min(), roof_points[:, 2].max()))
    if not (.008 < radius < .009 and .145 < lower < upper < .150):
        raise ValueError('Unexpected delivered marker dimensions')
    material = UsdShade.MaterialBindingAPI(roof[0]).ComputeBoundMaterial('physics')[0]
    if not material:
        raise ValueError('Missing cap physics material')
    for prim in roof:
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    end = UsdGeom.Cylinder.Define(stage, cap_path + '/Colliders/SimpleClosedEnd')
    end.CreateAxisAttr('Z')
    end.CreateRadiusAttr(radius)
    end.CreateHeightAttr(upper - lower)
    end.MakeMatrixXform().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, (lower + upper) / 2)))
    end.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(end.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdShade.MaterialBindingAPI.Apply(end.GetPrim()).Bind(material, materialPurpose='physics')
    end.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
    end.GetPrim().CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.0002)
    end.GetPrim().CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)

    # These invisible 0.1 g pads represent internal cap-neck friction. Letting
    # the robot squeeze a pad directly bypasses the cap shell and overconstrains
    # its prismatic joint. Only barrel contact belongs to this mechanism.
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
                 and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    damping = []
    for i in range(4):
        shoe_path = marker_path + '/Shoe' + str(i)
        shoe = stage.GetPrimAtPath(shoe_path)
        joint = stage.GetPrimAtPath(marker_path + '/RadialSpring' + str(i))
        if not shoe or not joint:
            raise ValueError('Missing delivered passive marker fit')
        excluded = [p.GetPath() for p in colliders
                    if not str(p.GetPath()).startswith((marker_path + '/Body/', shoe_path + '/'))]
        UsdPhysics.FilteredPairsAPI.Apply(shoe).CreateFilteredPairsRel().SetTargets(excluded)
        drive = UsdPhysics.DriveAPI(joint, 'linear')
        mass = float(UsdPhysics.MassAPI(shoe).GetMassAttr().Get())
        stiffness = float(drive.GetStiffnessAttr().Get())
        # Critical damping for the pad against its much heavier cap. Unlike
        # force/velocity clamping, this preserves static retention and pull-off.
        c = 2 * math.sqrt(stiffness * mass)
        drive.CreateDampingAttr(c)
        damping.append(c)
    return {
        'revision': 'l201-hollow-cap-v2',
        'cap_native_shapes_before': 241, 'cap_native_shapes_after': 17,
        'cap_box_envelope_shapes': 1,
        'closed_end_z_m': [lower, upper], 'outer_radius_m': radius,
        'cavity_walls_and_barrel_neck_unchanged': True,
        'shoe_contacts': 'barrel only', 'shoe_damping_n_s_m': damping,
        'spring_stiffness_travel_and_static_friction_unchanged': True,
        'visuals_mass_inertia_and_source_assets_unchanged': True,
        'limitation': 'Decorative cap-end grooves are omitted from collision geometry; retention is not force-calibrated to a measured pen.',
    }
