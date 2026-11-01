"""Transparent cleaning proxy: contact and sliding erase equal-area ink cells.

The ledger changes ink visibility only. It never commands or repositions the
robot/eraser, and its hidden state is used only for scoring, not model input.
"""

import numpy as np

from ...robots.transforms import pose_transform

INK_COLOR = (0.015, 0.015, 0.015)
INK_TEXT = "GPT6"
# Equal-area glyph cells are used by rendering and scoring together.
INK_ROWS = (
    "01111 11110 11111 01111",
    "10000 10001 00100 10000",
    "10000 10001 00100 10000",
    "10111 11110 00100 11110",
    "10001 10000 00100 10001",
    "10001 10000 00100 10001",
    "01111 10000 00100 01110",
)


def ink_centers():
    """Black GPT6 cells in board-local coordinates; board pose turns the text."""
    half_width = (len(INK_ROWS[0]) - 1) / 2
    return np.array(
        [
            ((col - half_width) * 0.008, (3 - row) * 0.008, 0.0102)
            for row, line in enumerate(INK_ROWS)
            for col, value in enumerate(line)
            if value == "1"
        ]
    )


class WipeLedger:
    def __init__(self, board_pose, centers=None, *, pad_z=0.018, half_extents=(0.0495, 0.0245), weights=None, facing_direction=-1):
        t = pose_transform(board_pose)
        self.centers = ink_centers() if centers is None else np.asarray(centers)
        self.world = self.centers @ t[:3, :3].T + t[:3, 3]
        self.normal = t[:3, 2]
        self.dirty = np.ones(len(self.centers), dtype=bool)
        self.pad_z, self.half_extents = pad_z, half_extents
        self.facing_direction = facing_direction
        self.weights = np.ones(len(self.centers)) if weights is None else np.asarray(weights)
        self.previous = None
        self.last_time = None

    def update(self, now, eraser_pose, *, pad_contact_n, held):
        if self.last_time is not None and now <= self.last_time:
            return ()  # Rendering/repeated scoring is not additional wiping.
        self.last_time = now
        t = pose_transform(eraser_pose)
        center = t[:3, 3] + self.pad_z * t[:3, 2]
        facing = float(t[:3, 2] @ self.normal) * self.facing_direction > 0.5
        active = held and pad_contact_n > 0.005 and facing
        old = self.previous
        self.previous = (center, active)
        if not active or old is None or not old[1]:
            return ()
        delta = center - old[0]
        slide = np.linalg.norm(delta - self.normal * (delta @ self.normal))
        if slide < 0.00005:
            return ()
        local = (self.world - t[:3, 3]) @ t[:3, :3]
        # Physical pad-board contact is independently required. The distance
        # tolerance only covers contact offsets and the visual ink's 0.2 mm lift.
        under_pad = (
            (np.abs(local[:, 0]) <= self.half_extents[0])
            & (np.abs(local[:, 1]) <= self.half_extents[1])
            & (np.abs(local[:, 2] - self.pad_z) <= 0.0015)
        )
        changed = np.flatnonzero(self.dirty & under_pad)
        self.dirty[changed] = False
        return tuple(map(int, changed))

    def metrics(self):
        initial = len(self.dirty)
        remaining = int(self.dirty.sum())
        return {
            "initial_cells": initial,
            "remaining_cells": remaining,
            "clean_fraction": 1.0 - float(self.weights[self.dirty].sum() / self.weights.sum()),
            "mechanism": "measured pad contact and sliding, equal-area visual cells",
        }


def score(e, rows):
    m = rows["eraser"]["wiping"]
    removed = m["initial_cells"] - m["remaining_cells"]
    required = (m["initial_cells"] if e.spec.parameters.get("require_all_writing")
                else e.spec.parameters["required_clean_cells"])
    return (
        removed >= required,
        {**m, "removed_cells": removed, "required_clean_cells": required,
         "release_required": False, "placement_required": False},
        "",
    )
