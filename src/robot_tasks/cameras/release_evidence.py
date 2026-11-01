"""Advisory release hypotheses from paired RGB-D; no evaluator/contact input."""
from dataclasses import dataclass
import numpy as np
from .depth import depth_array
from .geometry import project_world
from .grasp_evidence import visible_surfaces
from ..robots.transforms import pose_transform, pose_errors


def supported_points(frames, points, colours, tolerance=.0015):
    """Render agreement at projected world locations; occlusion is not release."""
    supported = np.zeros(len(points), dtype=bool)
    for frame in frames:
        c = frame.calibration
        camera = pose_transform(c['T_world_camera_pose_wxyz'])
        local = (points - camera[:3, 3]) @ camera[:3, :3]
        uv = project_world(points, c['T_world_camera_pose_wxyz'],
                           c['intrinsics_fx_fy_cx_cy'], c['distortion'])
        valid = np.isfinite(uv).all(axis=1) & (-local[:, 2] > 0)
        ids = np.flatnonzero(valid)
        pixels = np.rint(uv[ids]).astype(int)
        inside = (pixels >= 1).all(axis=1) & (pixels < [frame.width-1, frame.height-1]).all(axis=1)
        ids, pixels = ids[inside], pixels[inside]
        z = depth_array(frame)
        rgb = np.frombuffer(frame.data, np.uint8).reshape(frame.height, frame.width, 3)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                depth = z[pixels[:, 1]+dy, pixels[:, 0]+dx]
                colour = rgb[pixels[:, 1]+dy, pixels[:, 0]+dx].astype(float)
                supported[ids] |= (np.isfinite(depth) & (np.abs(depth + local[ids, 2]) < tolerance)
                                   & (np.linalg.norm(colour-colours[ids], axis=1) < 65))
    return supported


@dataclass
class ReleaseEvidence:
    reference_pose: tuple
    points: np.ndarray
    colours: np.ndarray
    reference_time_s: float
    probe_pose: tuple | None = None
    probe_points: np.ndarray | None = None
    probe_colours: np.ndarray | None = None
    probe_time_s: float | None = None

    @classmethod
    def capture(cls, observation, arm, table_z, profile):
        state = observation.robot[arm]
        surfaces = visible_surfaces(observation.rgb, state.tcp_pose_world_wxyz,
            state.gripper_width_m, table_z, narrow_grasp=True, robot_profile=profile)
        points = np.concatenate([s.points for s in surfaces]) if surfaces else np.empty((0, 3))
        colours = np.concatenate([s.colours for s in surfaces]) if surfaces else np.empty((0, 3))
        return cls(state.tcp_pose_world_wxyz, points, colours, observation.time_s)

    def update(self, observation, arm, table_z, profile, opening_complete):
        pose = observation.robot[arm].tcp_pose_world_wxyz
        result = dict(source='rgbd_and_measured_robot_only', advisory=True,
                      status='opening', reference_time_s=self.reference_time_s,
                      reason='Opening is not proof of release; inspect the prop and exit clearance.')
        if not opening_complete:
            return result
        # The object may fall during opening. Establish a NEW baseline after
        # opening, while the hand is still at its release pose; never compare
        # only against the pre-opening prop position.
        distance, angle = pose_errors(self.reference_pose, pose)
        if self.probe_pose is None:
            result['status'] = 'release_unconfirmed'
            if distance <= .002 and angle <= .03:
                after = self.capture(observation, arm, table_z, profile)
                self.probe_pose, self.probe_points, self.probe_colours = pose, after.points, after.colours
                self.probe_time_s = observation.time_s
            result['reason'] = 'Hand is open. Inspect settling; choose a small clear probe or another view. Detachment is unconfirmed.'
            return result
        distance, angle = pose_errors(self.probe_pose, pose)
        result.update(probe_translation_m=distance, probe_rotation_rad=angle,
                      baseline_points=len(self.probe_points))
        if distance < .002 and angle < .03:
            result['status'] = 'release_unconfirmed'
            result['reason'] = 'No separating motion observed; an open stationary hand cannot establish detachment.'
            return result
        before = pose_transform(self.probe_pose)
        current = pose_transform(pose)
        stationary = self.probe_points @ before[:3, :3].T + before[:3, 3]
        following = self.probe_points @ current[:3, :3].T + current[:3, 3]
        fixed_mask = supported_points(observation.rgb, stationary, self.probe_colours)
        moving_mask = supported_points(observation.rgb, following, self.probe_colours)
        fixed, moving = int(fixed_mask.sum()), int(moving_mask.sum())
        fixed_only = int(np.count_nonzero(fixed_mask & ~moving_mask))
        moving_only = int(np.count_nonzero(moving_mask & ~fixed_mask))
        count = max(len(self.probe_points), 1)
        result.update(stationary_points=fixed, following_points=moving,
                      stationary_exclusive_points=fixed_only, following_exclusive_points=moving_only)
        if fixed/count >= .4 and fixed_only >= 12 and fixed_only >= 3 * max(moving_only, 1):
            result.update(status='stationary_after_probe', reason='Visible prop surfaces remained near their post-opening location during the probe; inspect clearance and support before further withdrawal.')
        elif moving/count >= .4 and moving_only >= 12 and moving_only >= 3 * max(fixed_only, 1):
            result.update(status='object_following', reason='Visible surfaces followed the open hand: the prop may still be caught. Reassess before further withdrawal.')
        else:
            result.update(status='release_unknown', reason='Occlusion, motion or ambiguous surface matches prevent a release conclusion. Inspect from another view.')
        return result
