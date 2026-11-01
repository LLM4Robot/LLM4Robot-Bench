"""Explicit, recorded session overrides for the native orange FEM asset."""

import math


ATTRIBUTES = {
    "contact_offset_m": ("Collision", "physxCollision:contactOffset"),
    "rest_offset_m": ("Collision", "physxCollision:restOffset"),
    "dynamic_friction": ("PhysicsMaterial", "omniphysics:dynamicFriction"),
    "youngs_modulus_pa": ("PhysicsMaterial", "omniphysics:youngsModulus"),
    "poissons_ratio": ("PhysicsMaterial", "omniphysics:poissonsRatio"),
    "elasticity_damping": ("PhysicsMaterial", "physxDeformableMaterial:elasticityDamping"),
    "solver_iterations": ("", "physxDeformableBody:solverPositionIterationCount"),
}


def validate(values):
    unknown = set(values) - ATTRIBUTES.keys()
    if unknown:
        raise ValueError(f"Unknown orange physics settings: {sorted(unknown)}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Orange physics settings must be finite nonnegative numbers")
    if values["contact_offset_m"] <= values["rest_offset_m"]:
        raise ValueError("Orange contact offset must exceed rest offset")
    if values["youngs_modulus_pa"] <= 0 or not 0 <= values["poissons_ratio"] < 0.5:
        raise ValueError("Invalid orange elastic material")
    n = values["solver_iterations"]
    if n != int(n) or not 1 <= n <= 255:
        raise ValueError("Orange solver iterations must be an integer in [1, 255]")


def apply(stage, root, overrides):
    attributes = {
        key: stage.GetPrimAtPath(root + ("/" + child if child else "")).GetAttribute(name)
        for key, (child, name) in ATTRIBUTES.items()
    }
    if not all(attributes.values()):
        raise ValueError("Native orange FEM physics attributes are missing")
    native = {key: attr.Get() for key, attr in attributes.items()}
    # Existing task defaults, distinct from the locked source's native margins.
    effective = {**native, "contact_offset_m": 0.003, "rest_offset_m": 0.0015, **overrides}
    validate(effective)
    for key, attr in attributes.items():
        value = int(effective[key]) if key == "solver_iterations" else float(effective[key])
        if value != native[key]:
            attr.Set(value)
    return {
        "native": native,
        "effective": {key: attr.Get() for key, attr in attributes.items()},
        "requested_overrides": dict(overrides),
        "source_asset_modified": False,
        "material_calibrated_to_real_orange": False,
    }
