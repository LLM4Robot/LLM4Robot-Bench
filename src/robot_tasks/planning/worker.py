"""GPU worker. The same planner handles every task and both R5 arms."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Keep CUDA/JIT artifacts with the installed research environment, not the OS disk.
_cache = Path(os.environ.get("GPT_BENCHMARK_PLANNER_CACHE", str(Path(sys.prefix) / "cache")))

import numpy as np
from .assets import TCP_FRAME, build_assets
from ..robots.profile import load_robot_profile
from ..robots.control import bounded_measured_joints
from ..robots.transforms import (
    pose_transform,
    transform_pose,
    axis_rotation,
)
from .trajectory import JointPath
from .failures import PlanningFailure, diagnose_failure


def link_transforms(kin, positions):
    result = {kin.profile.base_link: np.eye(4)}
    transform = np.eye(4)
    for j, origin, q in zip(kin.chain, kin.origins, positions):
        rotation = np.eye(4)
        rotation[:3, :3] = axis_rotation(j.axis, q)
        transform = transform @ origin @ rotation
        result[j.child] = transform.copy()
    return result


class CuroboWorker:
    def __init__(self, profile, directory, attachments=None):
        for key, leaf in [
            ("XDG_CACHE_HOME", "xdg"),
            ("TORCH_EXTENSIONS_DIR", "torch"),
            ("WARP_CACHE_PATH", "warp"),
        ]:
            os.environ.setdefault(key, str(_cache / leaf))
        import torch
        import curobo.runtime
        import curobo._src.runtime

        # This pinned release copies scalar settings in its public re-export.
        # Set both before constructing compilers/planners.
        curobo.runtime.cache_dir = str(_cache / "curobo")
        curobo._src.runtime.cache_dir = curobo.runtime.cache_dir
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        self.torch = torch
        reference = os.environ.get("GPT_BENCHMARK_PLANNER_REFERENCE")
        if reference:
            from .reference_assets import build_reference_assets
            self.assets, self.kin = build_reference_assets(profile, directory, attachments, reference)
        else:
            self.assets, self.kin = build_assets(profile, directory, attachments)
        self.profile = profile
        self.planner = MotionPlanner(
            MotionPlannerCfg.create(
                robot=self.assets["config"],
                scene_model={"cuboid": {}},
                # Delivered room (126 fixed colliders) plus a 49-piece cup and
                # task fixtures exceed the previous 128-mesh capacity.
                collision_cache={"obb": 512, "mesh": 256},
                position_tolerance=0.001,
                orientation_tolerance=0.01,
                optimizer_collision_activation_distance=0.002,
                interpolation_dt=0.025,
                num_ik_seeds=32,
                num_trajopt_seeds=4,
                random_seed=0,
            )
        )
        self.metadata = self.assets["metadata"]
        import hashlib
        import importlib.metadata

        self.metadata.update(
            python=sys.executable,
            platform=sys.platform,
            library_version=importlib.metadata.version("nvidia-curobo"),
            torch_version=torch.__version__,
            gpu=torch.cuda.get_device_name(0),
            collision_geometry_sha256=hashlib.sha256(
                json.dumps(
                    {key: self.assets[key] for key in ("spheres", "finger_geometry", "wrist_spheres")},
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        )
        self.metadata.update(
            collision_scope="table, static fixtures, both R5 arms including wrist cameras/mounts; held-object clearance guard",
            payload_source="simulator contacts and surface geometry; planner-only privilege",
        )
        # Cross-check the generated model before allowing any real actuation.
        q = torch.tensor(
            [profile.mounts[0].home_positions_rad], device="cuda", dtype=torch.float32
        )
        pose = self.planner.kinematics.get_link_poses(q, [TCP_FRAME])
        actual = np.r_[
            pose.position.detach().cpu().numpy().reshape(-1, 3)[0],
            pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0],
        ]
        from ..robots.transforms import pose_errors

        if (
            max(
                pose_errors(
                    actual, self.kin.forward(q[0].cpu().numpy(), "left", "robot_base")
                )
            )
            > 1e-5
        ):
            raise RuntimeError("cuRobo FK differs from R5 URDF/TCP")

    def spheres(self, state, arm):
        rows = {k: list(v) for k, v in self.assets["spheres"].items()}
        end = list(self.assets["link6_static"])
        end.extend(self.assets.get("wrist_spheres", {}).get(arm, []))
        for (_, geometry), q in zip(
            self.assets["finger_geometry"].items(), state["finger_positions_m"]
        ):
            transform = np.array(geometry["origin"])
            transform[:3, 3] += transform[:3, :3] @ (np.array(geometry["axis"]) * q)
            for s in geometry["spheres"]:
                end.append(
                    {
                        "center": (
                            transform[:3, :3] @ s["center"] + transform[:3, 3]
                        ).tolist(),
                        "radius": s["radius"],
                    }
                )
        rows[self.profile.ee_link] = end
        return rows

    def world_spheres(self, arm, q, spheres):
        links = link_transforms(self.kin, q)
        result = []
        for link, rows in spheres.items():
            t = self.kin.bases[arm] @ links[link]
            for s in rows:
                if s["radius"] > 0:
                    result.append([*(t[:3, :3] @ s["center"] + t[:3, 3]), s["radius"]])
        return np.asarray(result)

    def world(self, arm, request):
        from copy import deepcopy
        from curobo.scene import Scene

        obstacles = deepcopy(request["scene"]["obstacles"])
        inv = np.linalg.inv(self.kin.bases[arm])
        for category in ("cuboid", "mesh"):
            for row in obstacles.get(category, {}).values():
                row["pose"] = list(transform_pose(inv @ pose_transform(row["pose"])))
        # Conservative boxes around the other arm's collision spheres. Named
        # meshes cache their vertices, so a moving mesh must never reuse a stale
        # geometry name; OBB poses can be updated safely within CUDA graph caches.
        for other, state in request["states"].items():
            if other == arm:
                continue
            rows = self.world_spheres(
                other, state["joint_positions_rad"], self.spheres(state, other)
            )
            for index, (x, y, z, r) in enumerate(rows):
                pose = np.eye(4)
                pose[:3, 3] = [x, y, z]
                obstacles.setdefault("cuboid", {})[f"other_{other}_{index}"] = {
                    "pose": list(transform_pose(inv @ pose)),
                    "dims": [2 * r] * 3,
                }
        return Scene.create(obstacles)

    def plan(self, request):
        from curobo.types import JointState, GoalToolPose
        from ..robots.transforms import pose_errors

        torch = self.torch
        paths, diagnostics = {}, {}
        states = request["states"]
        for command in request["commands"]:
            arm = command["arm"]
            phase_start = time.perf_counter()
            print(f"PLANNER_STAGE id={request['id']} arm={arm} stage=world", flush=True)
            state = states[arm]
            bounds = [
                next(x for x in self.profile.operating_limits if x.joint == n)
                for n in self.profile.joint_names
            ]
            # Keep the raw observation while bounding small PhysX constraint
            # settling errors, so a requested retreat remains plannable.
            reference = bounded_measured_joints(state["joint_positions_rad"], bounds)
            rows = self.spheres(state, arm)[self.profile.ee_link]
            spheres = torch.tensor(
                [[*s["center"], s["radius"]] for s in rows],
                device="cuda",
                dtype=torch.float32,
            )
            self.planner.kinematics.config.kinematics_config.update_link_spheres(
                self.profile.ee_link, spheres
            )
            self.planner.update_world(self.world(arm, request))
            print(f"PLANNER_STAGE id={request['id']} arm={arm} stage=solve world_s={time.perf_counter()-phase_start:.3f}", flush=True)
            solve_start = time.perf_counter()
            q0 = torch.tensor(
                [reference], device="cuda", dtype=torch.float32
            )
            start = JointState.from_position(
                q0, joint_names=list(self.profile.joint_names)
            )
            if command["kind"] == "cartesian_pose":
                world = self.kin.world_pose(command["values"], arm, command["frame"])
                local = transform_pose(
                    np.linalg.inv(self.kin.bases[arm]) @ pose_transform(world)
                )
                goal = GoalToolPose(
                    tool_frames=[TCP_FRAME],
                    position=torch.tensor(
                        local[:3], device="cuda", dtype=torch.float32
                    ).reshape(1, 1, 1, 1, 3),
                    quaternion=torch.tensor(
                        local[3:], device="cuda", dtype=torch.float32
                    ).reshape(1, 1, 1, 1, 4),
                )
                result = self.planner.plan_pose(goal, start, max_attempts=3)
            elif command["kind"] == "joint_position":
                goal = JointState.from_position(
                    torch.tensor(
                        [command["values"]], device="cuda", dtype=torch.float32
                    ),
                    joint_names=list(self.profile.joint_names),
                )
                result = self.planner.plan_cspace(goal, start, max_attempts=3)
            else:
                raise ValueError("Unsupported motion channel")
            contact_exit = None
            retreat = None
            if result is None or not bool(result.success.any()):
                diagnostic = diagnose_failure(
                    self.planner, arm, command["kind"], start, goal, result,
                    goal_distance_m=(float(np.linalg.norm(local[:3]))
                                     if command["kind"] == "cartesian_pose" else None),
                    reach_bound_m=(self.kin.reach_bound_m
                                   if command["kind"] == "cartesian_pose" else None),
                )
                if diagnostic.diagnostic["code"] != "start_collision":
                    raise diagnostic
                if request["scene"].get("allow_start_collision_retreat", False):
                    from .recovery import SceneOverlap, try_retreat
                    try:
                        diagnostic.diagnostic["start_overlap_pairs"] = SceneOverlap(self, arm, request).describe(reference)
                        q, retreat = try_retreat(self, arm, command, request, start, goal, diagnostic.diagnostic)
                        path = JointPath.from_waypoints(q, state["joint_positions_rad"], bounds)
                        path = path.stretched(max(path.duration_s, 1.,
                            float(np.sum(np.max(np.abs(np.diff(q, axis=0)), axis=1)) / .1)))
                    except (ValueError, RuntimeError) as exc:
                        diagnostic.diagnostic["retreat_rejection"] = str(exc)
                        raise diagnostic from exc
                elif command["kind"] == "cartesian_pose":
                    from .contact_exit import plan_contact_exit
                    try:
                        path, contact_exit = plan_contact_exit(self, arm, request, start, goal, world, bounds)
                    except (ValueError, RuntimeError) as exc:
                        diagnostic.diagnostic["contact_exit_rejection"] = str(exc)
                        raise diagnostic from exc
                else:
                    raise diagnostic
            else:
                trajectory = result.get_interpolated_plan()
                q = trajectory.position.detach().cpu().numpy().reshape(-1, len(trajectory.joint_names))
                indices = [trajectory.joint_names.index(n) for n in self.profile.joint_names]
                q = q[:, indices]
                path = JointPath.from_waypoints(q, state["joint_positions_rad"], bounds)
            if command["kind"] == "cartesian_pose":
                error = pose_errors(self.kin.forward(path.positions[-1], arm), world)
                if error[0] > 0.002 or error[1] > 0.02:
                    raise ValueError(f"{arm}: endpoint FK residual exceeds tolerance")
            paths[arm] = {"positions": path.positions, "times_s": path.times_s}
            diagnostics[arm] = {
                "waypoints": len(path.positions),
                **({"contact_exit": contact_exit} if contact_exit else {}),
                **({"start_collision_retreat": retreat} if retreat else {}),
                "duration_s": path.duration_s,
                "collision_checked": True,
                "payload_guard": "no held object",
                "start_joint_limit_correction_max_rad": max(
                    abs(q - r) for q, r in zip(state["joint_positions_rad"], reference)
                ),
                "solve_s": time.perf_counter() - solve_start,
            }
        # Check and execute the same time law. A common finish time avoids one
        # hand arriving early during a paired move; it is NOT a rigid attachment
        # or a guarantee of relative TCP pose along independently planned paths.
        duration = max(JointPath(**p).duration_s for p in paths.values())
        for arm, data in paths.items():
            path = JointPath(**data).stretched(duration)
            paths[arm] = {"positions": path.positions, "times_s": path.times_s}
            diagnostics[arm]["duration_s"] = path.duration_s
            diagnostics[arm]["timing_protocol"] = "shared_finish_preserved_time_law_v1"
            print(f"PLANNER_STAGE id={request['id']} arm={arm} stage=payload solve_s={diagnostics[arm]['solve_s']:.3f}", flush=True)
            self.check_payload(arm, path, request, diagnostics[arm])
            print(f"PLANNER_STAGE id={request['id']} arm={arm} stage=done", flush=True)
        if len(paths) > 1:
            self.check_bimanual(paths, states)
        return {"id": request["id"], "paths": paths, "diagnostics": diagnostics}

    def check_bimanual(self, paths, states):
        names = list(paths)
        trajectories = {
            a: JointPath(tuple(paths[a]["positions"]), tuple(paths[a]["times_s"]))
            for a in names
        }
        geometry = {a: self.spheres(states[a], a) for a in names}
        duration = max(t.duration_s for t in trajectories.values())
        for t in np.arange(0, duration + 0.01, 0.01):
            first, second = [
                self.world_spheres(a, trajectories[a].sample(t), geometry[a])
                for a in names
            ]
            distances = (
                np.linalg.norm(first[:, None, :3] - second[None, :, :3], axis=-1)
                - first[:, None, 3]
                - second[None, :, 3]
            )
            if np.any(distances < 0):
                raise ValueError(
                    "Bimanual planned trajectories overlap; neither arm was commanded"
                )

    def check_payload(self, arm, path, request, diagnostic):
        payloads = request["scene"].get("payloads", {}).get(arm, [])
        if not payloads:
            return
        import trimesh

        started = time.perf_counter()
        obstacles = request["scene"]["obstacles"]
        # JSON requests own immutable snapshots. Reuse static acceleration
        # structures, but invalidate if vertices, faces or poses have changed.
        if getattr(self, "_payload_obstacles", None) != obstacles:
            meshes = []
            for name, data in obstacles.get("mesh", {}).items():
                mesh = trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=True)
                mesh.apply_transform(pose_transform(data["pose"]))
                meshes.append((name, mesh))
            self._payload_obstacles, self._payload_meshes = obstacles, meshes
        meshes = self._payload_meshes
        mesh_queries = 0
        start = np.linalg.inv(
            pose_transform(
                self.kin.forward(request["states"][arm]["joint_positions_rad"], arm)
            )
        )
        for payload in payloads:
            # Measured held-body samples, never a fixed joint or object pose write.
            points = np.unique(np.asarray(payload["surface_points_world"]), axis=0)
            local = points @ start[:3, :3].T + start[:3, 3]
            for t in np.arange(0, path.duration_s + 0.02, 0.02):
                pose = pose_transform(self.kin.forward(path.sample(t), arm))
                world = local @ pose[:3, :3].T + pose[:3, 3]
                if world[:, 2].min() < request["scene"].get("table_surface_z_m", 0.0) - 0.002:
                    raise ValueError(
                        f"{arm}: carried-object clearance intersects table; lift before transport"
                    )
                low, high = world.min(axis=0), world.max(axis=0)
                for name, mesh in meshes:
                    # A point outside a closed mesh's AABB cannot be inside it.
                    # Expand by the existing tolerance; retain the exact narrow
                    # phase for every potentially penetrating point.
                    bounds = mesh.bounds
                    if np.any(high < bounds[0] - .001) or np.any(low > bounds[1] + .001):
                        continue
                    nearby = world[np.all((world >= bounds[0] - .001) & (world <= bounds[1] + .001), axis=1)]
                    if not len(nearby):
                        continue
                    mesh_queries += 1
                    # Negative outside; allow up to 1 mm contact tolerance, including final seating.
                    penetration = float(np.max(trimesh.proximity.signed_distance(mesh, nearby)))
                    if penetration > 0.001:
                        raise ValueError(
                            f"{arm}: carried-object path penetrates fixture {name} "
                            f"at t={t:.3f}s (depth={penetration * 1000:.3f} mm); "
                            "choose clearance then approach"
                        )
        diagnostic["payload_guard"] = (
            "held surface samples checked against table and fixture (1 mm contact tolerance)"
        )
        diagnostic.update(payload_check_s=time.perf_counter()-started, payload_mesh_queries=mesh_queries)


def main():
    import queue
    import threading

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--attachments", type=Path)
    args = parser.parse_args()
    incoming = queue.Queue()

    def read_requests():
        for line in sys.stdin:
            incoming.put(line)
        # Pipe ownership is the lifetime contract. In particular, killing the
        # Windows parent must not strand a WSL GPU process inside a long solve.
        os._exit(0)

    threading.Thread(target=read_requests, daemon=True).start()

    def emit(data):
        print("R5_PLANNER " + json.dumps(data, allow_nan=False), flush=True)

    try:
        worker = CuroboWorker(load_robot_profile(args.profile), args.directory,
                              json.loads(args.attachments.read_text(encoding="utf-8"))
                              if args.attachments else None)
        emit({"ready": True, **worker.metadata})
    except Exception as exc:
        import traceback

        traceback.print_exc()
        emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 2
    while True:
        line = incoming.get()
        request = {}
        try:
            request = json.loads(line)
            emit(worker.plan(request))
        except Exception as exc:
            import traceback

            traceback.print_exc()
            reply = {"id": request.get("id"), "error": f"{type(exc).__name__}: {exc}"}
            if isinstance(exc, PlanningFailure):
                reply["planning_failure"] = exc.diagnostic
            emit(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
