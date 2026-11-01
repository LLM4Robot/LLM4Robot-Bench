"""Validate short requested motions out of shallow contact, never ignore obstacles."""

import numpy as np

from ..robots.transforms import pose_transform

MAX_START_PENETRATION_M = 0.001
MAX_SPHERE_STEP_M = 0.00025
EXIT_CLEARANCE_M = 0.0005


def validate_clearances(clearances):
    """Each column is one sphere/obstacle pair; positive values mean separation."""
    gaps = np.asarray(clearances, dtype=float)
    if gaps.ndim != 2 or len(gaps) < 2 or not np.isfinite(gaps).all():
        raise ValueError("Invalid contact-exit clearance samples")
    penetration = np.maximum(-gaps, 0.)
    if not np.any(penetration[0] > 0):
        raise ValueError("No shallow contact to exit")
    if penetration[0].max() > MAX_START_PENETRATION_M:
        raise ValueError("Start contact exceeds 1 mm")
    # No new colliding pair, no worsening existing contact, and no re-entry.
    numerical = 1e-7
    if np.any(penetration[:, penetration[0] == 0] > numerical):
        raise ValueError("Contact exit introduces a new collision")
    if np.any(np.diff(penetration, axis=0) > numerical):
        raise ValueError("Contact exit increases penetration")
    if np.min(gaps[-1]) < EXIT_CLEARANCE_M:
        raise ValueError("Contact exit does not reach clearance")
    return float(penetration[0].max())


def scene_clearances(spheres, obstacles):
    """Exact sphere-to-mesh/OBB distances at dense samples, in world coordinates."""
    import trimesh

    shape = spheres.shape[:2]
    centres = spheres[..., :3].reshape(-1, 3)
    radius = spheres[..., 3].reshape(-1)
    columns = []
    for name, box in obstacles.get("cuboid", {}).items():
        inverse = np.linalg.inv(pose_transform(box["pose"]))
        local = centres @ inverse[:3, :3].T + inverse[:3, 3]
        q = np.abs(local) - np.asarray(box["dims"]) / 2
        distance = np.linalg.norm(np.maximum(q, 0.), axis=1) + np.minimum(q.max(axis=1), 0.)
        gap = (distance - radius).reshape(shape)
        # Another robot is not a passive surface from which to release contact.
        if name.startswith("other_") and np.any(gap < 0):
            raise ValueError("Contact exit intersects the other arm")
        columns.append(gap)
    for name, data in obstacles.get("mesh", {}).items():
        # Keep separately closed fixture parts separate: welding coincident
        # vertices between touching boxes creates non-manifold shared edges.
        mesh = trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)
        mesh.apply_transform(pose_transform(data["pose"]))
        low, high = mesh.bounds
        aabb_gap = np.linalg.norm(np.maximum(np.maximum(low - centres, centres - high), 0.), axis=1) - radius
        nearby = aabb_gap <= EXIT_CLEARANCE_M
        if not np.any(nearby):
            continue
        # Unknown/open surfaces cannot certify a contact exit.
        if not mesh.is_watertight:
            raise ValueError("Contact exit needs closed collision surfaces: " + name)
        gap = np.maximum(aabb_gap, EXIT_CLEARANCE_M + .001)
        gap[nearby] = -trimesh.proximity.signed_distance(mesh, centres[nearby]) - radius[nearby]
        columns.append(gap.reshape(shape))
    if not columns:
        raise ValueError("No collision surfaces available for contact exit")
    return np.concatenate(columns, axis=1)


def plan_contact_exit(worker, arm, request, start, goal, world, bounds):
    """Use only the requested nearby Cartesian endpoint and validate its full path."""
    from curobo.types import JointState
    from .failures import state_violations
    from .trajectory import JointPath
    from ..robots.transforms import pose_errors

    state = request["states"][arm]
    current = worker.kin.forward(state["joint_positions_rad"], arm)
    distance, angle = pose_errors(current, world)
    if len(request["commands"]) != 1 or not (0 < distance <= .03) or angle > .05:
        raise ValueError("Contact exit is limited to a single-arm short translation")
    if state_violations(worker.planner, start) != ["scene_collision"]:
        raise ValueError("Contact exit cannot relax other start constraints")
    geometry = worker.spheres(state, arm)
    geometry.pop(worker.profile.base_link, None)
    from copy import deepcopy
    obstacles = deepcopy(request["scene"]["obstacles"])
    for other, other_state in request["states"].items():
        if other == arm:
            continue
        for i, (x, y, z, r) in enumerate(worker.world_spheres(other,
                other_state["joint_positions_rad"], worker.spheres(other_state, other))):
            obstacles.setdefault("cuboid", {})[f"other_{other}_{i}"] = {
                "pose": [x, y, z, 1, 0, 0, 0], "dims": [2*r]*3}
    ik = worker.planner.ik_solver.solve_pose(goal, return_seeds=4, current_state=start)
    if ik.solution is None or not bool(ik.success.any()):
        raise ValueError("No collision-free endpoint IK for contact exit")
    solutions = ik.solution.detach().cpu().numpy().reshape(-1, len(bounds))
    valid = ik.success.detach().cpu().numpy().reshape(-1)
    candidates = [q for q, ok in zip(solutions, valid) if ok]
    candidates.sort(key=lambda q: np.linalg.norm(q - state["joint_positions_rad"]))
    rejections = []
    for endpoint in candidates:
        if np.max(np.abs(endpoint - state["joint_positions_rad"])) > .15:
            rejections.append("IK branch exceeds local joint-motion bound")
            continue
        try:
            # 0.25 mrad samples bound swept displacement; explicitly verify it.
            path = JointPath.from_waypoints([state["joint_positions_rad"], endpoint],
                state["joint_positions_rad"], bounds, max_step_rad=.00025)
            spheres = np.array([worker.world_spheres(arm, q, geometry) for q in path.positions])
            if np.max(np.linalg.norm(np.diff(spheres[..., :3], axis=0), axis=-1)) > MAX_SPHERE_STEP_M:
                raise ValueError("Contact-exit sampling too coarse")
            penetration = validate_clearances(scene_clearances(spheres, obstacles))
            # Keep joint limits, self-collision and every non-scene constraint.
            # After clearance, the original scene constraint must also pass.
            for i, q in enumerate(path.positions):
                js = JointState.from_position(worker.torch.tensor([q], device="cuda", dtype=worker.torch.float32),
                    joint_names=list(worker.profile.joint_names))
                violations = state_violations(worker.planner, js)
                if set(violations) - {"scene_collision"} or (i == len(path.positions)-1 and violations):
                    raise ValueError("Contact-exit path violates planner constraints")
            residual = pose_errors(worker.kin.forward(path.positions[-1], arm), world)
            if residual[0] > .001 or residual[1] > .01:
                raise ValueError("Contact-exit endpoint misses requested pose")
            # Slow release from contact; preserve checked interpolation and limits.
            path = path.stretched(max(path.duration_s, distance / .01, 1.))
            return path, {"protocol": "bounded_separating_contact_exit_v1",
                "initial_penetration_m": penetration, "samples": len(path.positions),
                "max_sphere_step_m": MAX_SPHERE_STEP_M, "endpoint_clearance_m": EXIT_CLEARANCE_M}
        except ValueError as exc:
            rejections.append(str(exc))
    raise ValueError("No validated separating path: " + "; ".join(rejections))
