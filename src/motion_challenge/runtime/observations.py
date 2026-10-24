"""Core-owned observation boundary, applied before adapter serialization."""
from dataclasses import fields

from motion_challenge.sdk import ProprioceptiveStatePacket

PRIVILEGED = 'privileged-state-v0'
PROPRIOCEPTIVE = 'ideal-proprioceptive-no-position-v1'


def validate_access(protocol, runtime):
    access = protocol.get('state_access', PRIVILEGED)
    if access not in (PRIVILEGED, PROPRIOCEPTIVE):
        raise ValueError('unknown observation access protocol')
    if runtime.get('state_access', PRIVILEGED) != access:
        raise ValueError('controller state_access differs from frozen protocol')
    return access


def controller_state(state, access):
    if access == PRIVILEGED:
        return state
    if access != PROPRIOCEPTIVE:
        raise ValueError('unknown observation access protocol')
    # Explicit allowlist: do not serialize the full packet and remove fields.
    return ProprioceptiveStatePacket(**{
        field.name: getattr(state, field.name)
        for field in fields(ProprioceptiveStatePacket) if field.name != 'schema'
    })
