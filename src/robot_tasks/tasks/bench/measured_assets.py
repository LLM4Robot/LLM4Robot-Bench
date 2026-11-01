"""User-confirmed SI calibration, applied to runtime layers, not historical runs.

2026-10-23: capped L2 marker 150 mm; unpackaged jelly target mass 26.4 g.
Marker diameter and radial cap retention are unchanged. Axial geometry, joint
anchors, COM, inertia and evaluation bounds use the same affine calibration.
"""
from copy import deepcopy

import numpy as np

MARKER_LENGTH_M = 0.150
MARKER_SOURCE_LENGTH_M = 0.14650000631809233
MARKER_SOURCE_CENTRE_Z_M = 0.07550000315904616
JELLY_MASS_KG = 0.0264
REVISION = "measured-marker-jelly-20261023"


def marker_axial(z):
    scale = MARKER_LENGTH_M / MARKER_SOURCE_LENGTH_M
    return MARKER_SOURCE_CENTRE_Z_M + (z - MARKER_SOURCE_CENTRE_Z_M) * scale


def measured_record(row, task_id):
    if task_id not in ("L2-01", "L2-06"):
        return row
    row = deepcopy(row)
    if task_id == "L2-01":
        for name in ("body", "cap"):
            for bound in row["tracked"][name]["bounds"]:
                bound[2] = marker_axial(bound[2])
    else:
        row["jelly"]["mass"] = JELLY_MASS_KG
    return row


def calibrate_marker(stage, path="/World/L2/marker"):
    """Run after contact simplification, before creating physics body views.

    Bake scaling below each rigid body: PhysX never sees a scaled rigid-body
    ancestor. This also scales both exterior collision proxies and the cavity.
    Root/body frames and the midpoint of the capped pen remain in place.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(path)
    if root.GetCustomDataByKey("measuredAssetsRevision") == REVISION:
        return {"revision": REVISION, "length_m": MARKER_LENGTH_M}
    scale = MARKER_LENGTH_M / MARKER_SOURCE_LENGTH_M
    stretch = np.diag([1., 1., scale, 1.])
    affine = stretch.copy()
    affine[2, 3] = marker_axial(0.)
    bodies = [stage.GetPrimAtPath(path + "/" + n)
              for n in ("Body", "Cap", "Shoe0", "Shoe1", "Shoe2", "Shoe3")]
    if not all(p and p.HasAPI(UsdPhysics.RigidBodyAPI) for p in bodies):
        raise ValueError("Measured marker requires the separate body, cap and four passive shoes")
    for body in bodies:
        shoe = body.GetName().startswith("Shoe")
        if shoe:
            xform = UsdGeom.Xformable(body)
            local = xform.GetLocalTransformation()
            p = local.ExtractTranslation()
            local.SetTranslateOnly(Gf.Vec3d(p[0], p[1], marker_axial(p[2])))
            xform.MakeMatrixXform().Set(local)
        # Passive 0.5 mm spherical pads retain their contact shape. Only their
        # axial location changes; a nonuniform sphere scale is unsupported by
        # PhysX and would silently change its collision radius.
        delta = np.eye(4) if shoe else affine
        for child in body.GetChildren():
            if child.IsA(UsdGeom.Xformable):
                xform = UsdGeom.Xformable(child)
                local = np.asarray(xform.GetLocalTransformation()).T
                xform.MakeMatrixXform().Set(Gf.Matrix4d((delta @ local).T.tolist()))
        mass = UsdPhysics.MassAPI(body)
        com = np.asarray(mass.GetCenterOfMassAttr().Get(), dtype=float)
        if np.isfinite(com).all():
            com = (delta @ np.r_[com, 1.])[:3]
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
        inertia = np.asarray(mass.GetDiagonalInertiaAttr().Get(), dtype=float)
        if np.all(inertia > 0):
            # Principal axes are aligned with the authored marker frame.
            q = mass.GetPrincipalAxesAttr().Get()
            if abs(abs(q.GetReal()) - 1.) > 1e-6:
                raise ValueError("Audit rotated marker principal inertia before calibrating")
            moments = inertia.sum() / 2 - inertia
            moments[2] *= scale * scale
            inertia = moments.sum() - moments
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
    for i in range(4):
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(path + f"/RadialSpring{i}"))
        for attr, delta in ((joint.GetLocalPos0Attr(), affine),
                            (joint.GetLocalPos1Attr(), np.eye(4))):
            p = (delta @ np.r_[np.asarray(attr.Get()), 1.])[:3]
            attr.Set(Gf.Vec3f(*map(float, p)))
    # PhysX capsules cannot represent ellipsoidal end caps. Keep the native
    # radius and encode the calibrated axial extent in the cylinder height,
    # cancelling the inherited axial scale on this analytic collider.
    nib = UsdGeom.Capsule(stage.GetPrimAtPath(path + '/Body/Colliders/Nib'))
    if not nib or nib.GetAxisAttr().Get() != 'Z':
        raise ValueError('Expected the native Z-axis nib capsule')
    radius = nib.GetRadiusAttr().Get()
    nib.CreateHeightAttr(scale * (nib.GetHeightAttr().Get() + 2 * radius) - 2 * radius)
    local = np.asarray(nib.GetLocalTransformation()).T
    local = local @ np.diag([1., 1., 1./scale, 1.])
    nib.MakeMatrixXform().Set(Gf.Matrix4d(local.T.tolist()))
    root.SetCustomDataByKey("measuredAssetsRevision", REVISION)
    return {"revision": REVISION, "length_m": MARKER_LENGTH_M,
            "axial_scale": scale, "diameter_scale": 1.,
            "midpoint_preserved": True, "mass_and_radial_spring_settings_preserved": True}


def calibrate_jelly(stage, path="/World/L2/jelly/Jelly"):
    """Explicit total FEM mass overrides the source material's density mass."""
    from pxr import UsdPhysics
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError("Missing jelly FEM mesh")
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(JELLY_MASS_KG)
    return {"revision": REVISION, "mass_kg": JELLY_MASS_KG,
            "basis": "User-confirmed net mass; explicit FEM total mass",
            "shape_elasticity_friction_and_render_material_preserved": True}
