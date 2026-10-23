"""Metrics independent of controller rewards, with explicit failure coverage."""

from dataclasses import dataclass
import math

from .reference import normalize_quaternion


def joint_mae_rad(actual, reference):
    if not actual or len(actual) != len(reference):
        raise ValueError("joint vectors must be nonempty and have equal length")
    if not all(math.isfinite(x) for x in (*actual, *reference)):
        raise ValueError("non-finite joint values")
    # Bounded G1 joints: no periodic wrap that could hide a wrong revolution.
    return math.fsum(abs(a-b) for a, b in zip(actual, reference)) / len(actual)


def orientation_error_rad(actual_xyzw, reference_xyzw):
    a, b = normalize_quaternion(actual_xyzw), normalize_quaternion(reference_xyzw)
    dot = min(1.0, abs(math.fsum(x*y for x, y in zip(a, b))))
    return 2.0 * math.acos(dot)


def mpjpe_m(actual_positions, reference_positions):
    if not actual_positions or len(actual_positions) != len(reference_positions):
        raise ValueError("body positions must be nonempty and have equal length")
    errors = []
    for a, b in zip(actual_positions, reference_positions):
        if len(a) != 3 or len(b) != 3 or not all(math.isfinite(x) for x in (*a, *b)):
            raise ValueError("body positions require finite xyz")
        errors.append(math.sqrt(math.fsum((x-y)**2 for x, y in zip(a, b))))
    return math.fsum(errors) / len(errors)


@dataclass(frozen=True)
class EpisodeOutcome:
    clip_id: str
    planned_steps: int
    observed_steps: int
    first_failure_step: int | None

    def __post_init__(self):
        if self.planned_steps < 1 or not 0 <= self.observed_steps <= self.planned_steps:
            raise ValueError("invalid episode coverage")
        if self.first_failure_step is not None and not 1 <= self.first_failure_step <= self.observed_steps:
            raise ValueError("failure step must identify an observed post-physics transition")

    @property
    def success(self):
        return self.first_failure_step is None and self.observed_steps == self.planned_steps

    @property
    def completion(self):
        # A failing transition does not count as successfully completed.
        steps = self.observed_steps if self.first_failure_step is None else self.first_failure_step-1
        return steps / self.planned_steps


def summarize(outcomes, expected_clip_ids):
    expected = tuple(expected_clip_ids)
    outcomes = tuple(outcomes)
    actual = [outcome.clip_id for outcome in outcomes]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected clip manifest must be nonempty and unique")
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("missing, extra or duplicate clip results; cannot rank")
    if any(o.first_failure_step is None and o.observed_steps < o.planned_steps for o in outcomes):
        raise ValueError("incomplete infrastructure run; cannot rank")
    return {
        "clips": len(outcomes),
        "success_count": sum(o.success for o in outcomes),
        "success_rate": sum(o.success for o in outcomes) / len(outcomes),
        "macro_completion_fraction": math.fsum(o.completion for o in outcomes)/len(outcomes),
    }
