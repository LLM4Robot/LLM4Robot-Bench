"""Caches with explicit evidence keys and short lifetimes, never simulator renders."""

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json


class JointReadCache:
    """Share measured positions only while the physics engine is not advancing.

    Installed on an articulation instance; outside the controller callback all
    reads go straight to Isaac. Unknown signatures and non-array results bypass
    caching. Cached arrays are copied so consumers cannot poison later reads.
    """
    def __init__(self, timings):
        self.timings = timings
        self.active = False
        self.values = {}

    @contextmanager
    def scope(self):
        if self.active:
            raise RuntimeError("Joint read cache cannot nest physics callbacks")
        self.values.clear()
        self.active = True
        try:
            yield
        finally:
            self.active = False
            self.values.clear()

    def reader(self, name, original):
        def read(*args, **kwargs):
            import numpy as np
            if not self.active or args or set(kwargs) != {"joint_indices"}:
                return original(*args, **kwargs)
            indices = kwargs["joint_indices"]
            if indices is None:
                return original(*args, **kwargs)
            key = (name, tuple(indices))
            if key in self.values:
                self.timings.counted("joint_read_hits")
                return self.values[key].copy()
            result = original(*args, **kwargs)
            self.timings.counted("joint_read_misses")
            if isinstance(result, np.ndarray):
                self.values[key] = result.copy()
            return result
        return read


class FrameEncodingCache:
    """Byte-identical PNG preparation for repeated observations/retries.

    Key includes RGB bytes, dimensions, calibration, measured robot state,
    annotation geometry and resize setting. No observation text, timestamp,
    history, depth data or model result is reused. One cache per episode.
    """
    def __init__(self, original, timings, *, max_bytes=32 * 1024 * 1024, max_entries=6):
        self.original, self.timings = original, timings
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.bytes = 0

    @staticmethod
    def _json(value):
        if is_dataclass(value):
            return asdict(value)
        raise TypeError("Nonstandard annotation metadata; bypass cache")

    def __call__(self, frame, max_edge, *, robot=None, bench=None):
        with self.timings.measure("model.image_prepare"):
            try:
                metadata = json.dumps((frame.width, frame.height, max_edge,
                    frame.calibration, robot, bench), default=self._json,
                    allow_nan=False, separators=(",", ":")).encode()
                digest = hashlib.sha256(metadata)
                digest.update(frame.data)
                key = digest.digest()
            except (TypeError, ValueError):
                return self.original(frame, max_edge, robot=robot, bench=bench)
            if key in self.entries:
                self.timings.counted("image_encoding_hits")
                self.entries.move_to_end(key)
                return deepcopy(self.entries[key][0])
            self.timings.counted("image_encoding_misses")
            result = self.original(frame, max_edge, robot=robot, bench=bench)
            size = len(result[0]) + len(metadata)
            if size <= self.max_bytes and self.max_entries > 0:
                while self.entries and (self.bytes + size > self.max_bytes or len(self.entries) >= self.max_entries):
                    _, (_, removed) = self.entries.popitem(last=False)
                    self.bytes -= removed
                self.entries[key] = (deepcopy(result), size)
                self.bytes += size
            return result
