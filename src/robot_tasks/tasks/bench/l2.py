"""Maintained L2 tasks use the version-locked L20 deliveries."""

from .native_l2_specs import books, drawer, marker, orange
from .pearl_pouring import pouring
from .jelly import jelly
from .native_l3_specs import wiping

BUILDERS = {
    "L2-01": marker,
    "L2-02": books,
    "L2-03": pouring,
    "L2-04": orange,
    "L2-05": drawer,
    "L2-06": jelly,
    "L2-07": wiping,
}
