"""Simulator-independent contracts. SI units except USD joint angular limits (degrees)."""

from dataclasses import dataclass, field

from ..l1.specs import Design


@dataclass(frozen=True)
class Joint:
    name: str
    body0: str  # Empty means a world anchor.
    body1: str
    anchor: tuple[float, float, float]
    # Missing axes are locked. (lower, upper, stiffness, damping); angular degrees.
    axes: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchSpec:
    design: Design
    arms: tuple[str, ...] = ("left",)
    joints: tuple[Joint, ...] = ()
    mechanism: str = "rigid_contacts"
    limitations: str = (
        "Engineering dimensions and friction; not calibrated to physical props."
    )
    facts: str = ""
    parameters: dict = field(default_factory=dict)

    @property
    def observers(self):
        return tuple(arm for arm in ("left", "right") if arm not in self.arms)


def build_spec(task_id):
    from . import l1, l2, l3

    for module in (l1, l2, l3):
        if task_id in module.BUILDERS:
            return module.BUILDERS[task_id]()
    raise ValueError(f"No local scene implementation for {task_id}")
