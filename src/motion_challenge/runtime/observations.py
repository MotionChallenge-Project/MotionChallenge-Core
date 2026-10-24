"""Core-owned observation boundary, applied before adapter serialization."""
from dataclasses import fields

from motion_challenge.sdk import ProprioceptiveStatePacket

PROPRIOCEPTIVE = 'ideal-proprioceptive-no-position-v1'


def validate_access(protocol, runtime):
    access = protocol.get('state_access')
    if access != PROPRIOCEPTIVE:
        raise ValueError('protocol must declare ideal proprioception without position')
    if runtime.get('state_access') != access:
        raise ValueError('controller state_access differs from frozen protocol')
    return access


def controller_state(state, access):
    if access != PROPRIOCEPTIVE:
        raise ValueError('unknown observation access protocol')
    # Explicit allowlist: do not serialize the full packet and remove fields.
    return ProprioceptiveStatePacket(**{
        field.name: getattr(state, field.name)
        for field in fields(ProprioceptiveStatePacket) if field.name != 'schema'
    })
