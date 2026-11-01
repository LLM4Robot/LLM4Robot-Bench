"""Declared assistance and measurement boundaries for the original six L2 tasks."""

# Level classification does not select execution or scoring semantics. Wiping
# moved to L2-07 while retaining its existing contact-wear protocol.
NATIVE_PROTOCOL_TASKS = frozenset({
    "L2-01", "L2-02", "L2-03", "L2-04", "L2-05", "L2-06",
})


def uses_native_l2_protocol(task_id):
    return task_id in NATIVE_PROTOCOL_TASKS

PROTOCOL = {
    "prompt_version": "guided_visual_l2_v3",
    "scoring_version": "native_l2_validity_v2",
    "scope": "guided visual closed-loop manipulation with declared prop/grasp hints",
    "policy_information": "RGB, camera calibration, robot state, execution feedback, public task facts",
    "privileged_information": "object/FEM/contact truth for scoring and planner clearance only",
    "control": "pose/joint and gripper-opening targets; fixed controller gains and timing, no force command",
    "model_wait": "physics paused during model requests",
}

def protocol_metadata(config) -> dict:
    return {**PROTOCOL, "policy_information": (
        ("RGB-D" if config.include_depth else "RGB")
        + ", camera calibration, robot state, execution feedback, public task facts")}


OPERATING_NOTES = """
Shared L2 protocol (guided_visual_l2_v3):
The task supplies prop dimensions and grasp/direction hints. Use current camera views
and robot feedback to locate and manipulate the actual objects; these hints
are not measured current object poses. This is a guided execution test, not
an unassisted affordance-discovery test.
You command poses and gripper openings. There is no direct force, torque or
motion-speed command. Smaller pose increments and intermediate gripper openings
are available; their effect on contact must be checked, not assumed. The fixed
controller executes motions between observations; physics pauses while you think.
For a paired transport, command both TCPs in the same action with compatible
displacements. Shared execution timing does not guarantee a level load or grip.
An observation-only arm may reposition its camera but must not touch task props
or their fixtures, even with its wrist or camera housing.
There is no in-episode reset. A confirmed irreversible process violation ends
the episode. Numerical simulation failures are recorded separately from your
task outcome. Do not infer an unavailable force sensor from the scoring rules.
"""
