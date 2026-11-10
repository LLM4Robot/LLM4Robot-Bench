"""Bounded aggregate timings, including failures and nested/threaded work."""

from collections import defaultdict
from contextlib import contextmanager
import threading
import time


class Timings:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.started = clock()
        self.local = threading.local()
        self.lock = threading.Lock()
        self.phases = {}
        self.counters = defaultdict(int)

    @contextmanager
    def measure(self, name):
        stack = getattr(self.local, "stack", None)
        if stack is None:
            stack = self.local.stack = []
        entry = [self.clock(), 0.0]
        stack.append(entry)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = max(0.0, self.clock() - entry[0])
            stack.pop()
            if stack:
                stack[-1][1] += elapsed
            with self.lock:
                row = self.phases.setdefault(name, dict(calls=0, failures=0,
                    inclusive_s=0.0, exclusive_s=0.0, max_s=0.0))
                row["calls"] += 1
                row["failures"] += int(failed)
                row["inclusive_s"] += elapsed
                row["exclusive_s"] += max(0.0, elapsed - entry[1])
                row["max_s"] = max(row["max_s"], elapsed)

    def counted(self, name, count=1):
        with self.lock:
            self.counters[name] += count

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)
        return measured

    def snapshot(self):
        with self.lock:
            return {
                "schema_version": 1,
                "elapsed_s": max(0.0, self.clock() - self.started),
                "scope": "episode runner entry through latest checkpoint; excludes pre-run assembly and final backend teardown",
                "accounting": "inclusive phases nest; exclusive removes same-thread children; model and video threads overlap, so do not sum phases as wall time",
                "phases": {key: dict(value) for key, value in sorted(self.phases.items())},
                "counters": dict(sorted(self.counters.items())),
            }
