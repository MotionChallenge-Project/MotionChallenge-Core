"""Optional adapter utility: reconstruct FK using permitted measurements only.

The returned legacy-shaped packet has a synthetic, fixed origin. It is not a
measurement or estimate of global position. Adapters must disclose its use and
prove observation/action equivalence or an author-supported deployment path.
"""
import copy
from functools import lru_cache
from pathlib import Path

import mujoco
from scipy.spatial.transform import Rotation

from motion_challenge.sdk import ProprioceptiveStatePacket
from motion_challenge.simulation.physics import World, forward_kinematics


@lru_cache(maxsize=4)
def _template(profile):
    return World(profile)


class KinematicStateView:
    def __init__(self, robot_profile, origin=(0.0, 0.0, 0.0)):
        # FK reads the canonical model; only its per-controller MjData mutates.
        self.world = copy.copy(_template(str(Path(robot_profile).resolve())))
        self.world.data = mujoco.MjData(self.world.model)
        self.origin = tuple(float(x) for x in origin)
        if len(self.origin) != 3:
            raise ValueError('synthetic origin must have three coordinates')

    def reconstruct(self, state):
        if not isinstance(state, ProprioceptiveStatePacket):
            raise TypeError('kinematic view requires the restricted state packet')
        world = self.world
        if tuple(state.joint_names) != world.joint_names:
            raise ValueError('kinematic view joint order differs from robot profile')
        data = world.data
        data.qpos[:] = world.model.qpos0
        data.qpos[:3] = self.origin
        q = state.root_orientation_xyzw
        data.qpos[3:7] = (q[3], *q[:3])
        data.qpos[world.qadr] = state.joint_position_rad
        data.qvel[:] = 0
        data.qvel[:3] = state.root_linear_velocity_world_m_s
        data.qvel[3:6] = Rotation.from_quat(q).inv().apply(state.root_angular_velocity_world_rad_s)
        data.qvel[world.vadr] = state.joint_velocity_rad_s
        data.time = state.time_s
        forward_kinematics(world.model, data)
        return world.state()
