"""Native photo writing, removed only where the held pad actually wipes."""

import numpy as np

from .native_l2 import NativeL2Scene
from .wiping import WipeLedger


def native_ink(asset_directory):
    """One-millimetre samples of the delivered mask; weights preserve ink area."""
    from PIL import Image

    mask = np.asarray(Image.open(asset_directory / "ink_mask.png"), dtype=float) / 255
    rgb = np.asarray(Image.open(asset_directory / "writing.png"), dtype=float) / 255
    height, width = mask.shape
    centers, weights, colors = [], [], []
    for v in range(0, height, 5):
        for u in range(0, width, 5):
            tile = mask[v:v+5, u:u+5]
            weight = float(tile.sum())
            if weight < 0.5:
                continue
            centers.append(((u + 2.5) / width * 0.386 - 0.193,
                            0.128 - (v + 2.5) / height * 0.256, 0.0101))
            weights.append(weight)
            pixels = rgb[v:v+5, u:u+5]
            color = (pixels * tile[..., None]).sum(axis=(0, 1)) / weight
            colors.append(np.where(color <= 0.04045, color / 12.92,
                                   ((color + 0.055) / 1.055) ** 2.4))
    return np.array(centers), np.array(weights), np.array(colors)


class WipingScene(NativeL2Scene):
    root = "/World/L3Wiping"

    def __init__(self, backend, config, spec):
        from isaacsim.core.prims import RigidPrim
        from pxr import UsdGeom, UsdShade, Sdf

        super().__init__(backend, config, spec)
        if spec.parameters.get("whiteboard_surface"):
            material = UsdShade.Material.Define(self.stage, self.root + "/WhiteboardMaterial")
            shader = UsdShade.Shader.Define(self.stage, self.root + "/WhiteboardMaterial/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.92, 0.92, 0.92))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            surface = self.stage.GetPrimAtPath(self.root + "/board/Visuals/WritingSurface")
            UsdShade.MaterialBindingAPI.Apply(surface).Bind(material)
        self.centers, self.weights, self.colors = native_ink(
            self.deliveries["green_board_20261008"][0].parent)
        # Replace only the runtime writing surface with the same delivered mask.
        # The immutable source asset and its original texture remain untouched.
        UsdGeom.Imageable(self.stage.GetPrimAtPath(
            self.root + "/board/Visuals/PhotoWriting")).MakeInvisible()
        self.ink = UsdGeom.Mesh.Define(self.stage, self.root + "/board/RemovableInk")
        offsets = np.array([[-.0005, -.0005, 0], [.0005, -.0005, 0],
                            [.0005, .0005, 0], [-.0005, .0005, 0]])
        self.ink.CreatePointsAttr((self.centers[:, None, :] + offsets).reshape(-1, 3).tolist())
        self.ink.CreateSubdivisionSchemeAttr("none")
        self.ink.CreateDoubleSidedAttr(True)
        self.ink.CreateDisplayColorPrimvar("uniform")
        self.pad_contact = RigidPrim(
            self.paths["eraser"], name="wiping_pad_contacts", reset_xform_properties=False,
            track_contact_forces=True,
            contact_filter_prim_paths_expr=[self.root + "/board/Colliders/WritingSurface"],
            max_contact_count=2048, disable_stablization=False)
        self._new_ledger()
        self.backend.scene_report["household"]["writing"] = {
            "source": "delivered ink_mask.png and writing.png", "sample_pitch_m": .001,
            "cells": len(self.centers), "removal": "one contact pass; no repeat-count threshold"}

    def _new_ledger(self):
        p = self.spec.parameters
        self.ledger = WipeLedger(p["board_pose"], self.centers, weights=self.weights,
                                 pad_z=p["pad_z"], half_extents=p["pad_half_extents"])
        self._render_ink()

    def _render_ink(self):
        alive = np.flatnonzero(self.ledger.dirty)
        self.ink.GetFaceVertexCountsAttr().Set([4] * len(alive))
        self.ink.GetFaceVertexIndicesAttr().Set((alive[:, None] * 4 + np.arange(4)).ravel().tolist())
        self.ink.GetDisplayColorPrimvar().Set(self.colors[alive].tolist())

    def reset(self, seed):
        super().reset(seed)
        self.pad_contact.initialize()
        self._new_ledger()

    def measure(self):
        rows = super().measure()
        row = rows["eraser"]
        contacts = np.asarray(self.pad_contact.get_contact_force_matrix(dt=self.config.physics_dt_s))
        if contacts.shape != (1, 1, 3) or not np.isfinite(contacts).all():
            raise ValueError("Invalid eraser/board contact measurement")
        changed = self.ledger.update(
            self.backend.time_s(), row["pose"],
            pad_contact_n=float(np.linalg.norm(contacts[0, 0])),
            held=min(row["fingers"][self.spec.arms[0]]) > 0.05)
        if changed:
            self._render_ink()
        row["wiping"] = self.ledger.metrics()
        return rows
