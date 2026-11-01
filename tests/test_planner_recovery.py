"""Recovery must never trade one collision for another or permit deep overlap."""
import unittest
import numpy as np
from robot_tasks.planning.recovery import check_overlap_trace, try_retreat


class RetreatTraceTests(unittest.TestCase):
    def test_shallow_monotonic_exit(self):
        check_overlap_trace([[.0007,0],[.0004,0],[0,0]])

    def test_new_pair_is_rejected_even_when_global_max_decreases(self):
        with self.assertRaisesRegex(ValueError,'new'):
            check_overlap_trace([[.002,0],[.001,.0005],[0,0]])

    def test_later_collision_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'deepens'):
            check_overlap_trace([[.002],[.0005],[.001],[0]])

    def test_small_repeated_increases_cannot_accumulate(self):
        with self.assertRaisesRegex(ValueError,'deepens'):
            check_overlap_trace([[.001],[.001015],[.00103],[0]])

    def test_deep_start_rejected(self):
        with self.assertRaisesRegex(ValueError,'5 mm'):
            check_overlap_trace([[.0051],[0]])

    def test_uncleared_endpoint_rejected(self):
        with self.assertRaisesRegex(ValueError,'endpoint'):
            check_overlap_trace([[.001],[.0001]])

    def test_nan_and_negative_rejected(self):
        for d in ([[np.nan],[0]],[[-.001],[0]]):
            with self.assertRaisesRegex(ValueError,'Invalid'):
                check_overlap_trace(d)

    def test_no_contact_is_not_an_escape(self):
        with self.assertRaisesRegex(ValueError,'No measurable'):
            check_overlap_trace([[0],[0]])

    def test_disabled_scene_never_enters_recovery(self):
        with self.assertRaisesRegex(ValueError,'not enabled'):
            try_retreat(None,'right',{}, {'scene':{}},None,None,{})

    def test_physical_collision_never_enters_planner_only_recovery(self):
        with self.assertRaisesRegex(ValueError,'Physical'):
            try_retreat(None,'right',{}, {'scene':{'allow_start_collision_retreat':True,
                'unexpected_contact_blocked':True}},None,None,{})

    def test_self_collision_cannot_be_waived(self):
        with self.assertRaisesRegex(ValueError,'self/limit'):
            try_retreat(None,'right',{}, {'scene':{'allow_start_collision_retreat':True}},
                None,None,{'violated_constraints':['scene_collision','self_collision']})

    def test_payload_retreat_is_rejected(self):
        request={'scene':{'allow_start_collision_retreat':True,'payloads':{'right':[{}]}},
                 'commands':[{}]}
        with self.assertRaisesRegex(ValueError,'carried'):
            try_retreat(None,'right',{'kind':'cartesian_pose'},request,None,None,
                        {'violated_constraints':['scene_collision']})


if __name__ == '__main__':
    unittest.main()
