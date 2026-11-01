"""Measured rigid collision feedback, independent of task success scoring.

Fingers may contact props. Non-finger links/attachments must not be driven into
props or furniture, and neither hand may be driven into the other robot.
Explicitly marked compliant proxies may deform within strict force/deformation bounds.
This stops an action, not an episode; the policy still chooses any recovery.
"""

import math


def unexpected_pair(collider0, collider1, mounts):
    def robot_part(path):
        for arm, root in mounts.items():
            if path.startswith(root + "/link"):
                link = path[len(root) + 1:].split("/")[0]
                if link in {f"link{i}" for i in range(1, 9)}:
                    return arm, link
        return None

    parts = [robot_part(p) for p in (collider0, collider1)]
    if parts[0] and parts[1] and parts[0][0] == parts[1][0]:
        return None  # Articulation's self-collision settings govern own links.
    for i, part in enumerate(parts):
        if part and (part[1] not in {"link7", "link8"} or parts[1-i]):
            return {"arm": part[0], "link": part[1],
                    "component": "wrist_camera_or_mount" if "/WristSensor/" in (collider0, collider1)[i] else part[1],
                    "robot_collider": (collider0, collider1)[i],
                    "other_collider": (collider0, collider1)[1-i]}
    return None


class CollisionMonitor:
    def __init__(self, stage, profile, *, allow_actor_recovery=False):
        self.allow_actor_recovery = allow_actor_recovery
        from pxr import PhysicsSchemaTools, PhysxSchema, Sdf
        from omni.physx import get_physx_simulation_interface

        self.stage = stage
        self.mounts = {m.name: m.prim_path for m in profile.mounts}
        self.robot_actors = {PhysicsSchemaTools.sdfPathToInt(Sdf.Path(f"{root}/link{i}"))
                             for root in self.mounts.values() for i in range(1, 9)}
        self.reset()
        for root in self.mounts.values():
            for i in range(1, 9):
                PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(f"{root}/link{i}")).CreateThresholdAttr(0.)
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._contacts)

    def reset(self):
        self.pending = []
        self.compliant_limits = {}
        self.blocked = None
        self.peak = None
        self.steps = 0
        self.events = 0
        self.recovery_contacts = {}
        self.recovery_elapsed_s = 0.
        self.recovery_window_s = .25
        self.release_recovery = False
        self.release_actor_contacts = {}
        self.recovery_actor_contacts = {}
        self.active_actor_contacts = {}

    @staticmethod
    def _pair(row):
        return row.get("robot_collider"), row.get("other_collider")

    def _contacts(self, headers, data):
        from pxr import PhysicsSchemaTools

        for header in headers:
            # Hundreds of pearl/pearl contacts are irrelevant to arm feedback.
            if (not header.num_contact_data or
                (header.actor0 not in self.robot_actors and header.actor1 not in self.robot_actors)):
                continue
            row = unexpected_pair(str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                                  str(PhysicsSchemaTools.intToSdfPath(header.collider1)), self.mounts)
            if row is None:
                continue
            points = data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]
            impulse = sum(math.sqrt(sum(float(v)**2 for v in p.impulse)) for p in points)
            separation = min(float(p.separation) for p in points)
            robot_first = header.actor0 in self.robot_actors
            robot_actor, other_actor = ((header.actor0,header.actor1) if robot_first
                                       else (header.actor1,header.actor0))
            self.pending.append({**row, "impulse_ns": impulse, "separation_m": separation,
                "robot_actor": str(PhysicsSchemaTools.intToSdfPath(robot_actor)),
                "other_actor": str(PhysicsSchemaTools.intToSdfPath(other_actor))})

    def _compliant_limits(self, path):
        """Only explicitly marked, physically compliant external shapes opt in.

        This models bounded deformation of a soft collision proxy, not a
        general exemption for leaf names, low forces or existing contacts.
        Inter-arm, unmarked and rigid-material contacts retain rigid guards.
        """
        cache=self.compliant_limits
        if path in cache:
            return cache[path]
        limits=None
        stage=getattr(self,'stage',None)
        external=not any(path.startswith(root+'/') for root in getattr(self,'mounts',{}).values())
        if stage is not None and external:
            from pxr import UsdShade,UsdPhysics
            prim=stage.GetPrimAtPath(path)
            if prim and prim.HasAPI(UsdPhysics.CollisionAPI):
                deformation=prim.GetAttribute('task:compliantContactDeformationLimit')
                force=prim.GetAttribute('task:compliantContactForceLimit')
                if deformation and force and deformation.Get() is not None and force.Get() is not None:
                    depth,cap=float(deformation.Get()),float(force.Get())
                    material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')[0]
                    stiffness=(material.GetPrim().GetAttribute('physxMaterial:compliantContactStiffness')
                               if material else None)
                    if (math.isfinite(depth) and math.isfinite(cap) and 0<depth<=.01+1e-8
                            and 0<cap<=.05+1e-8 and stiffness and stiffness.Get() is not None
                            and math.isfinite(stiffness.Get()) and stiffness.Get()>0):
                        schemas=material.GetPrim().GetMetadata('apiSchemas')
                        if schemas and 'PhysxMaterialAPI' in schemas.GetAddedOrExplicitItems():
                            limits=(min(cap,.05),min(depth,.01))
        cache[path]=limits
        return limits

    def after_step(self, dt):
        self.steps += 1
        self.recovery_elapsed_s += dt
        current = {}
        for row in self.pending:
            row["force_n"] = row["impulse_ns"] / dt
            if not all(math.isfinite(row[k]) for k in ("force_n", "separation_m")):
                from ..types import InvalidSimulationState
                raise InvalidSimulationState("Nonfinite rigid collision report")
            if self.peak is None or row["force_n"] > self.peak["force_n"]:
                self.peak = row
            actor_pair = row.get('robot_actor'),row.get('other_actor')
            if all(actor_pair) and (actor_pair not in current or row['force_n']>current[actor_pair]['force_n']):
                current[actor_pair]=dict(row)
            compliant=self._compliant_limits(row.get('other_collider',''))
            force_limit,depth_limit=compliant or (.1,.002)
            if compliant:
                row.update(contact_model='bounded_compliant',force_limit_n=force_limit,
                           deformation_limit_m=depth_limit)
            # Noise/contact margin threshold, NOT a prop damage or task-failure
            # criterion. Real camera force sensing is not assumed or claimed.
            if row["force_n"] >= force_limit or row["separation_m"] < -depth_limit:
                if compliant:
                    # The soft-contact cap is strict, including recovery. Never
                    # inherit the larger known-rigid-contact recovery allowance.
                    self.events += 1
                    if self.blocked is None or row['force_n']>self.blocked['force_n']:
                        self.blocked=row
                    continue
                old = self.recovery_contacts.get(self._pair(row))
                if old is None and getattr(self, "allow_actor_recovery", False):
                    # Adjacent leaf/petal colliders belong to the same rigid
                    # body. During bounded release/retreat recovery, treat those
                    # faces as the existing actor contact, not new obstacles.
                    old = self.recovery_actor_contacts.get(actor_pair)
                if (old is not None and self.recovery_elapsed_s <= self.recovery_window_s
                    and row["separation_m"] >= max(-.002, old["separation_m"] - .0005)
                    and row["force_n"] <= max(old["force_n"] + .1,
                        2. if self.release_recovery else 0.)):
                    # A new planned motion may leave an already reported contact.
                    # Release may briefly increase the old contact's load as
                    # the held prop falls free. The arm remains stationary;
                    # new bodies, deeper penetration and the force cap still stop.
                    continue
                self.events += 1
                if self.blocked is None or row["force_n"] > self.blocked["force_n"]:
                    self.blocked = row
        self.pending.clear()
        self.active_actor_contacts = current

    def begin_action(self, *, allow_retreat=False, release_arm=None):
        row = self.blocked
        if row is None and allow_retreat and self.release_recovery:
            # Hand opening can finish with the released prop still lightly
            # touching the wrist. Hand off only CURRENT contacts to the short
            # retreat guard; never revive stale contacts or its 2 s grace.
            active=[self.active_actor_contacts[k] for k in self.release_actor_contacts
                    if k in self.active_actor_contacts]
            if active:row=max(active,key=lambda r:r['force_n'])
        # A release-only action holds every arm joint and only opens one hand.
        # Give that opening time to complete against the already-known pair;
        # new contacts, excessive forces and deeper penetration still stop it.
        release = bool(getattr(self, "allow_actor_recovery", False) and row
                       and release_arm is not None and release_arm == row.get('arm'))
        self.recovery_contacts = ({self._pair(row): dict(row)}
                                  if (allow_retreat or release) and row else {})
        self.recovery_elapsed_s = 0.
        self.recovery_window_s = 2. if release else .25
        self.release_recovery = release
        self.release_actor_contacts = ({(row['robot_actor'],row['other_actor']):dict(row)}
            if release and row.get('robot_actor') and row.get('other_actor') else {})
        self.recovery_actor_contacts = ({(row['robot_actor'],row['other_actor']):dict(row)}
            if (release or allow_retreat) and row and row.get('robot_actor') and row.get('other_actor') else {})
        self.blocked = None

    def metadata(self):
        return {"scope": "non-finger rigid links/cameras/mounts against external shapes; inter-arm contact including fingers",
                "normal_finger_prop_contacts_allowed": True,
                "policy_feedback_source": "binary simulated PhysX contact flag; detailed impulses kept in execution log, no calibrated hardware force sensor",
                "response": "cancel current motion, hold measured pose and grip, return collision_blocked; no automatic retreat or task failure",
                "force_noise_floor_n": .1, "penetration_guard_m": .002,
                "compliant_contact_rule": "Opt-in collider attributes AND positive physical material compliance; at most 10 mm deformation and 0.05 N, stricter force cap, no recovery override. Inter-arm and every unmarked rigid contact retain original guards.",
                "retreat_contact_grace_s": .25, "retreat_max_extra_penetration_m": .0005,
                "actor_release_recovery_enabled": getattr(self, "allow_actor_recovery", False),
                "release_only_contact_grace_s": 2. if getattr(self, "allow_actor_recovery", False) else 0.,
                "release_known_contact_force_floor_cap_n": 2.,
                "release_rule": "hold all arm joints and open one hand; existing rigid actor pair only (including adjacent faces), no extra penetration, force bounded by max(previous + 0.1 N, 2 N); new bodies still stop",
                "retreat_rule": "existing rigid actor pair during a new planned motion, without growing force or deeper penetration; new bodies still stop. Only fresh release contacts can hand off to this 0.25 s guard",
                "limitations": "PhysX rigid contacts only; FEM orange contact is not reported by this sensor. Own-arm contacts follow the existing self-collision setting."}

    def close(self):
        self.subscription = None
