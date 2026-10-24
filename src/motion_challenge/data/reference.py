"""Deterministic reference windows, without model-specific retargeting."""

from bisect import bisect_right
from dataclasses import dataclass
import math

from motion_challenge.sdk import ReferenceFrame, Vector


def normalize_quaternion(q: Vector) -> Vector:
    if len(q) != 4 or not all(math.isfinite(x) for x in q):
        raise ValueError("quaternion must contain four finite xyzw values")
    norm = math.sqrt(sum(x*x for x in q))
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    return tuple(x / norm for x in q)


def slerp(a: Vector, b: Vector, fraction: float) -> Vector:
    a, b = normalize_quaternion(a), normalize_quaternion(b)
    dot = sum(x*y for x, y in zip(a, b))
    if dot < 0:
        b, dot = tuple(-x for x in b), -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion(tuple(x + fraction*(y-x) for x, y in zip(a, b)))
    angle = math.acos(dot)
    return tuple((math.sin((1-fraction)*angle)*x + math.sin(fraction*angle)*y)
                 / math.sin(angle) for x, y in zip(a, b))


@dataclass(frozen=True)
class Motion:
    joint_names: tuple[str, ...]
    times_s: tuple[float, ...]
    joint_positions_rad: tuple[Vector, ...]
    root_positions_world_m: tuple[Vector, ...]
    root_orientations_xyzw: tuple[Vector, ...]

    def __post_init__(self):
        n = len(self.times_s)
        if n < 2 or len(set(self.joint_names)) != len(self.joint_names) or not self.joint_names:
            raise ValueError("motion requires at least two samples and unique joint names")
        if self.times_s[0] != 0 or not all(math.isfinite(t) for t in self.times_s):
            raise ValueError("motion time must start at zero and remain finite")
        if any(b <= a for a, b in zip(self.times_s, self.times_s[1:])):
            raise ValueError("sample times must strictly increase")
        for values, width in ((self.joint_positions_rad, len(self.joint_names)),
                              (self.root_positions_world_m, 3),
                              (self.root_orientations_xyzw, 4)):
            if len(values) != n or any(len(row) != width for row in values):
                raise ValueError("inconsistent motion shape")
            if not all(math.isfinite(x) for row in values for x in row):
                raise ValueError("non-finite motion value")
        if any(abs(sum(x*x for x in q)-1) > 1e-3 for q in self.root_orientations_xyzw):
            raise ValueError("quaternion norm outside tolerance")

    @property
    def duration_s(self) -> float:
        return self.times_s[-1]

    def sample(self, requested_time_s: float) -> ReferenceFrame:
        if not math.isfinite(requested_time_s):
            raise ValueError("non-finite sample time")
        t = min(self.duration_s, max(0.0, requested_time_s))
        lo = max(0, min(len(self.times_s)-2, bisect_right(self.times_s, t)-1))
        hi = lo + 1
        alpha = (t-self.times_s[lo])/(self.times_s[hi]-self.times_s[lo])
        def interpolate(values):
            return tuple(a + alpha*(b-a) for a, b in zip(values[lo], values[hi]))
        return ReferenceFrame(requested_time_s, t, interpolate(self.joint_positions_rad),
                              interpolate(self.root_positions_world_m),
                              slerp(self.root_orientations_xyzw[lo], self.root_orientations_xyzw[hi], alpha))


def reference_window(motion: Motion, time_s: float, offsets_s: tuple[float, ...],
                     max_preview_s: float, max_reference_history_s: float = 0.0) -> tuple[ReferenceFrame, ...]:
    """Kernel constructs only the approved window; never hands Motion to a worker."""
    if not math.isfinite(max_preview_s) or max_preview_s < 0:
        raise ValueError("invalid preview limit")
    if not math.isfinite(max_reference_history_s) or max_reference_history_s < 0:
        raise ValueError("invalid reference history limit")
    if not offsets_s or any(not math.isfinite(x) or x < -max_reference_history_s or x > max_preview_s for x in offsets_s):
        raise ValueError("reference offset exceeds declared preview permission")
    if any(b <= a for a, b in zip(offsets_s, offsets_s[1:])):
        raise ValueError("reference offsets must strictly increase")
    return tuple(motion.sample(time_s + offset) for offset in offsets_s)
