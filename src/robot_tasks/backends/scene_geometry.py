"""Known static bench geometry shared by rendering, notes and image annotations."""

from dataclasses import dataclass

BENCH_POSITION = (0.25, 0.0, -0.03)
BENCH_SIZE = (1.5, 1.65, 0.06)
BENCH_TOP = BENCH_POSITION[2] + BENCH_SIZE[2] / 2
# Legacy recordings contain a rendered TCP sphere. Kept only for their grasp
# evidence exclusion; the current backend no longer renders a marker.
TCP_MARKER_RADIUS_M = 0.004


@dataclass(frozen=True)
class BenchGeometry:
    centre: tuple
    size: tuple
    prim_path: str
    world_axes: str

    @property
    def top(self):
        return self.centre[2] + self.size[2] / 2


def bench_geometry(config=None):
    """Public environment geometry, never task object or target coordinates."""
    if config is not None and config.raw.get("simulation", {}).get("environment") == "tuojing_lab_v4":
        if config.raw.get("task", {}).get("scene_delivery") == "video_20261015":
            return BenchGeometry((0.04354960845762923, 1.242650417122201, 0.8675),
                                 (1.59, 0.79, 0.025), "/World/Furniture/Worktable/Top",
                                 "+X overview right, +Y toward robots, +Z up; room-floor origin")
        return BenchGeometry((0.04354960845762923, 1.187650417122201, 0.7675),
                             (1.7, 0.9, 0.025), "/World/Furniture/Worktable/Top",
                             "+X windows, +Y desk wall, +Z up; origin on room floor")
    return BenchGeometry(BENCH_POSITION, BENCH_SIZE, "/World/Bench",
                         "+Z up; +X forward from the configured bases; +Y toward the left base")
