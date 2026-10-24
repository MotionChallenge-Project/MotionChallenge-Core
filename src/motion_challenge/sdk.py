"""Transport-neutral single-environment API; arrays use named joint order.

These frozen value objects are an API contract, NOT a security sandbox.
The adapter runs in the evaluator process; no simulator handles are supplied.
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
    # Internal physics/scoring state; never supplied to controllers.
    # Adapter FK may also construct this geometry container at a synthetic origin.


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
    def reset(self, initial_state: ProprioceptiveStatePacket,
              reference: tuple[ReferenceFrame, ...], seed: int) -> None:
        """Clear all histories/caches before each clip; no clip name is supplied."""
        ...

    def act(self, state: ProprioceptiveStatePacket,
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
    def reset_batch(self, slots: tuple[int, ...], states: tuple[ProprioceptiveStatePacket, ...],
                    references: tuple[tuple[ReferenceFrame, ...], ...],
                    seeds: tuple[int, ...]) -> None:
        ...

    def act_batch(self, slots: tuple[int, ...], states: tuple[ProprioceptiveStatePacket, ...],
                  references: tuple[tuple[ReferenceFrame, ...], ...]
                  ) -> tuple[ControlCommand, ...]:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class StateBatch:
    """Read-only NumPy views. First axis follows slots; copy retained histories."""
    joint_names: tuple[str, ...]
    time_s: object                         # [N]
    joint_position_rad: object             # [N, J]
    joint_velocity_rad_s: object           # [N, J]
    root_orientation_xyzw: object          # [N, 4]
    root_linear_velocity_world_m_s: object # [N, 3]
    root_angular_velocity_world_rad_s: object # [N, 3]


@dataclass(frozen=True)
class ReferenceBatch:
    """Read-only views of the same approved references, [N, K, ...]."""
    requested_time_s: object
    source_time_s: object
    joint_position_rad: object
    root_position_world_m: object
    root_orientation_xyzw: object
    joint_velocity_rad_s: object
    root_linear_velocity_world_m_s: object
    root_angular_velocity_world_rad_s: object
    body_names: tuple[str, ...]
    body_position_world_m: object           # [N, K, B, 3]
    body_orientation_xyzw: object           # [N, K, B, 4]
    body_linear_velocity_world_m_s: object  # [N, K, B, 3]
    body_angular_velocity_world_rad_s: object # [N, K, B, 3]


@dataclass(frozen=True)
class ControlBatch:
    """Each field is a finite [N, J] array in the supplied joint order."""
    joint_position_target_rad: object
    joint_velocity_target_rad_s: object
    kp_nm_rad: object
    kd_nm_s_rad: object
    feedforward_torque_nm: object


class ArrayBatchController(Protocol):
    """Optional array_batch_entrypoint: no per-sample packet reconstruction.

    Views expire after each call; copy anything needed for history. Slots can
    be reordered or partially reset. Engine, device, padding and sub-batching
    remain entirely adapter-owned, as for BatchController.
    """
    def reset_batch(self, slots: tuple[int, ...], states: StateBatch,
                    references: ReferenceBatch, seeds: tuple[int, ...]) -> None: ...
    def act_batch(self, slots: tuple[int, ...], states: StateBatch,
                  references: ReferenceBatch) -> ControlBatch: ...
    def close(self) -> None: ...
