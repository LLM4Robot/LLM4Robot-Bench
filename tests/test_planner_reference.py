import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robot_tasks.registry import resolve_task
from robot_tasks.planning.reference_assets import build_reference_assets, geometry_digest

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'assets/r5/planner_reference/geometry.json'

class PlannerReferenceTests(unittest.TestCase):
    def test_relocation_preserves_collision_geometry_and_rejects_changed_input(self):
        _,config=resolve_task('L2-01')
        reference=json.loads(REFERENCE.read_text());expected=reference['robot']
        with tempfile.TemporaryDirectory() as tmp, patch(
            'robot_tasks.planning.reference_assets.geometry_digest',
            return_value=reference['inputs']['attachments_sha256_10um'],
        ) as digest:
            result,_=build_reference_assets(config.robot,Path(tmp),{},REFERENCE)
            for key in ('spheres','finger_geometry','link6_static','wrist_spheres'):
                self.assertEqual(result[key],expected[key])
            arm=Path(result['config']['robot_cfg']['kinematics']['urdf_path'])
            self.assertTrue(arm.is_file())
            self.assertNotIn('package://',arm.read_text())
            digest.return_value='changed geometry'
            with self.assertRaisesRegex(ValueError,'does not match'):
                build_reference_assets(config.robot,Path(tmp),{},REFERENCE)

    def test_geometry_identity_ignores_transform_roundoff_but_detects_mm_change(self):
        a={'right':[{'vertices':[[.123456,0.,-.0023]]}]}
        b={'right':[{'vertices':[[.12345600001,-0.,-.0023]]}]}
        self.assertEqual(geometry_digest(a),geometry_digest(b))
        b['right'][0]['vertices'][0][0]+=.001
        self.assertNotEqual(geometry_digest(a),geometry_digest(b))

if __name__=='__main__':unittest.main()
