"""Native L3 task props."""

from .native_l3_specs import bottle, sweeping
from .push_triangle import build as pushing

BUILDERS = {
    "L3-01": bottle,
    "L3-03": pushing,
    "L3-04": sweeping,
}
