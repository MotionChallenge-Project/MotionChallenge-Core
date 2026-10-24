"""Transport-neutral single-environment API; arrays use named joint order.

These frozen value objects are an API contract, NOT a security sandbox.
The development worker separates processes, but is NOT an OS security sandbox.
All units are SI; quaternions are xyzw; world is right-handed, z-up.
"""

from dataclasses import dataclass
from typing import Protocol

Vector = tuple[float, ...]


@dataclass(frozen=True)
class BodyState:
    name: str
    position_world_m: Vector
    orientation_xyzw: Vector
    linear_velocity_world_m_s: Vector
    angular_velocity_world_rad_s: Vector


@dataclass(frozen=True)
class ReferenceFrame:
    # source_time is clamped at clip endpoints, requested_time is not.
    requested_time_s: float
    source_time_s: float
    joint_position_rad: Vector
    root_position_world_m: Vector
    root_orientation_xyzw: Vector
    joint_velocity_rad_s: Vector = ()
    root_linear_velocity_world_m_s: Vector = ()
    root_angular_velocity_world_rad_s: Vector = ()
    bodies: tuple[BodyState, ...] = ()


@dataclass(frozen=True)
class StatePacket:
    time_s: float
    joint_names: tuple[str, ...]
    joint_position_rad: Vector
    joint_velocity_rad_s: Vector
    root_position_world_m: Vector
    root_orientation_xyzw: Vector
    root_linear_velocity_world_m_s: Vector
    root_angular_velocity_world_rad_s: Vector
    bodies: tuple[BodyState, ...] = ()
    # Fixed public state schema v0 is explicitly privileged. Deployable/state
    # access tiers must use separate packet schemas and result tracks later.


@dataclass(frozen=True)
class ProprioceptiveStatePacket:
    """Ideal measurements; no translation, contacts, or actual body poses.

    Velocities are measured at the root body origin and expressed in world
    axes. Full orientation includes yaw. Local FK is permitted; integrating
    measurements into displacement or a root-position estimate is not.
    """

    time_s: float
    joint_names: tuple[str, ...]
    joint_position_rad: Vector
    joint_velocity_rad_s: Vector
    root_orientation_xyzw: Vector
    root_linear_velocity_world_m_s: Vector
    root_angular_velocity_world_rad_s: Vector
    schema: str = 'ideal-proprioceptive-no-position-v1'


@dataclass(frozen=True)
class ControlCommand:
    """PD command in packet joint order. Kernel owns motor limits and dynamics.

    tau = kp*(q_target-q) + kd*(qd_target-qd) + tau_ff, evaluated each physics
    step. Controller-side filtering is part of the submitted controller.
    """

    joint_position_target_rad: Vector
    joint_velocity_target_rad_s: Vector
    kp_nm_rad: Vector
    kd_nm_s_rad: Vector
    feedforward_torque_nm: Vector


class Controller(Protocol):
    def reset(self, initial_state: StatePacket,
              reference: tuple[ReferenceFrame, ...], seed: int) -> None:
        """Clear all histories/caches before each clip; no clip name is supplied."""
        ...

    def act(self, state: StatePacket,
            reference: tuple[ReferenceFrame, ...]) -> ControlCommand:
        """No simulator handle, reset request, termination flag or score access."""
        ...

    def close(self) -> None:
        ...


class BatchController(Protocol):
    """Optional adapter interface; slot IDs identify independent controller state.

    Calls may contain any nonempty subset, in any order. Reset only the supplied
    slots. Return commands in the supplied order. No clip identity or score is
    exposed. Adapters own padding, model-specific microbatches and cache layout.
    """
    def reset_batch(self, slots: tuple[int, ...], states: tuple[StatePacket, ...],
                    references: tuple[tuple[ReferenceFrame, ...], ...],
                    seeds: tuple[int, ...]) -> None:
        ...

    def act_batch(self, slots: tuple[int, ...], states: tuple[StatePacket, ...],
                  references: tuple[tuple[ReferenceFrame, ...], ...]
                  ) -> tuple[ControlCommand, ...]:
        ...

    def close(self) -> None:
        ...
