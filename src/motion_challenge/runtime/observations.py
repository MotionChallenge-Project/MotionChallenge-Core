"""Core-owned explicit observation allowlists; legacy results retain their schema."""
from dataclasses import fields

from motion_challenge.sdk import (
    ProprioceptiveStatePacket, DeployableStatePacket, DeployableStateBatch,
)

PROPRIOCEPTIVE = 'ideal-proprioceptive-no-position-v1'
DEPLOYABLE = 'proprioceptive-no-position-no-linear-velocity-v1'
SUPPORTED_ACCESS = (PROPRIOCEPTIVE, DEPLOYABLE)


def validate_access(protocol, runtime):
    access = protocol.get('state_access')
    if access not in SUPPORTED_ACCESS:
        raise ValueError('unknown observation access protocol')
    if runtime.get('state_access') != access:
        raise ValueError('controller state_access differs from frozen protocol')
    return access


def controller_state(state, access):
    if access not in SUPPORTED_ACCESS:
        raise ValueError('unknown observation access protocol')
    packet = DeployableStatePacket if access == DEPLOYABLE else ProprioceptiveStatePacket
    return packet(**{
        field.name: getattr(state, field.name)
        for field in fields(packet) if field.name != 'schema'
    })


def controller_arrays(state, access):
    """Strip linear velocity before scalar, packet-batch or tensor adapters."""
    if access == DEPLOYABLE:
        return DeployableStateBatch(**{
            field.name: getattr(state, field.name) for field in fields(DeployableStateBatch)
        })
    if access == PROPRIOCEPTIVE:
        return state
    raise ValueError('unknown observation access protocol')


def controller_references(reference, access):
    """Strict adapters receive source root/joint motion, never scoring FK."""
    from motion_challenge.sdk import MotionReferenceBatch
    if access == DEPLOYABLE:
        return MotionReferenceBatch(**{
            field.name: getattr(reference, field.name) for field in fields(MotionReferenceBatch)
        })
    if access == PROPRIOCEPTIVE:
        return reference
    raise ValueError('unknown observation access protocol')
