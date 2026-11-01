"""Reject invalid soft-body settings before authoring any USD overrides."""

import unittest

from robot_tasks.tasks.bench.orange_physics import validate


class OrangePhysicsTests(unittest.TestCase):
    def setUp(self):
        self.values = {
            "contact_offset_m": .003,
            "rest_offset_m": .0015,
            "dynamic_friction": .45,
            "youngs_modulus_pa": 35000,
            "poissons_ratio": .42,
            "elasticity_damping": .005,
            "solver_iterations": 128,
        }

    def test_valid_native_and_finite_stiffer_material(self):
        validate(self.values)
        validate({**self.values, "youngs_modulus_pa": 100000})

    def test_invalid_properties_are_rejected(self):
        for changes in (
            {"dynamic_friction": -1},
            {"dynamic_friction": float("nan")},
            {"youngs_modulus_pa": float("inf")},
            {"youngs_modulus_pa": 0},
            {"poissons_ratio": .5},
            {"contact_offset_m": .0015},
            {"rest_offset_m": .004},
            {"solver_iterations": 1.5},
            {"solver_iterations": 0},
            {"solver_iterations": 256},
            {"solver_iterations": True},
            {"mass_kg": .001},
            {"static_friction": 10},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate({**self.values, **changes})


if __name__ == "__main__":
    unittest.main()
