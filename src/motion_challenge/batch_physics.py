"""Official MuJoCo batch stepping with independent controller/episode slots.

The coordinator alone touches Batch. Slot threads keep the serial evaluator's
clock, reference cache, history and worker protocol unchanged. Each barrier
coalesces their next reset/advance; finished slots leave the barrier. PD is
recomputed on every physics substep, including fractional endpoint steps.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import threading

import mujoco
from mujoco import batch
import numpy as np

from .physics import World, command_values


@dataclass
class Request:
    op: str
    values: object = None
    until: float = 0.0
    done: threading.Event = field(default_factory=threading.Event)


class BatchWorld(World):
    def __init__(self, profile, pool, slot):
        super().__init__(profile)
        self.pool, self.slot = pool, slot

    def reset(self, motion):
        super().reset(motion)
        self.pool.submit(self.slot, Request('reset'))
        return self.state()

    def advance(self, command, until_s):
        self.pool.submit(self.slot, Request('advance', command_values(command, len(self.joints)), until_s))


class BatchPool:
    def __init__(self, profile, size, threads=None):
        if size < 1 or (threads is not None and threads < 1):
            raise ValueError('batch size and threads must be positive')
        self.worlds = [BatchWorld(profile, self, i) for i in range(size)]
        self.batch = batch.Batch(self.worlds[0].model, nsim=size,
                                 nthread=threads or size, persistent=True)
        self.qpos, self.qvel, self.ctrl = (self.batch.bind(n) for n in ('qpos', 'qvel', 'ctrl'))
        self.time = self.batch.bind('time')
        self.dt = self.batch.expand('opt.timestep')
        self.condition = threading.Condition()
        self.pending, self.active, self.error = {}, set(), None

    def submit(self, slot, request):
        with self.condition:
            if self.error:
                raise RuntimeError('batch evaluation aborted') from self.error
            self.pending[slot] = request
            self.condition.notify_all()
            self.condition.wait_for(lambda: request.done.is_set() or self.error is not None)
            if self.error:
                raise RuntimeError('batch evaluation aborted') from self.error

    def _execute(self, requests):
        resets = sorted(i for i, r in requests.items() if r.op == 'reset')
        if resets:
            self.batch.reset(ids=resets, stage=mujoco.mjtStage.mjSTAGE_NONE)
            for i in resets:
                self.qpos[i] = self.worlds[i].data.qpos
                self.qvel[i] = self.worlds[i].data.qvel
            self.dt[resets] = self.worlds[0].profile['physics_dt_s']
            self.batch.forward(ids=resets)
        advances = sorted(i for i, r in requests.items() if r.op == 'advance')
        if advances:
            world = self.worlds[0]
            target = np.array([requests[i].until for i in advances])
            values = np.array([requests[i].values for i in advances])
            while True:
                mask = self.time[advances] < target-1e-10
                if not mask.any():
                    break
                ids = np.asarray(advances)[mask]
                q, qd, kp, kd, ff = values[mask].transpose(1, 0, 2)
                self.dt[ids] = np.minimum(world.profile['physics_dt_s'], target[mask]-self.time[ids])
                torque = kp*(q-self.qpos[np.ix_(ids, world.qadr)]) + kd*(qd-self.qvel[np.ix_(ids, world.vadr)]) + ff
                self.ctrl[ids] = np.clip(torque, -world.limits, world.limits)
                previous_time = self.time[ids].copy()
                expected_time = previous_time + self.dt[ids]
                self.batch.step(ids=ids, nstep=1)
                if not np.allclose(self.time[ids], expected_time, rtol=0, atol=1e-10):
                    raise FloatingPointError('MuJoCo reset or discontinuous batch simulation time')
                if not np.isfinite(self.qpos[ids]).all() or not np.isfinite(self.qvel[ids]).all():
                    raise FloatingPointError('non-finite batch physics state')
            # Match World.advance: restore dt BEFORE the final mj_forward.
            self.dt[advances] = world.profile['physics_dt_s']
            self.batch.forward(ids=advances)
        for i in requests:
            self.batch.get_data(i, self.worlds[i].data)

    def run(self, function):
        """Run function(slot, world) once per slot; errors wake every waiter."""
        self.active = set(range(len(self.worlds)))
        def work(i):
            try:
                return function(i, self.worlds[i])
            except BaseException as exc:
                with self.condition:
                    self.error = self.error or exc
                    self.condition.notify_all()
                raise
            finally:
                with self.condition:
                    self.active.discard(i)
                    self.condition.notify_all()
        with ThreadPoolExecutor(max_workers=len(self.worlds)) as executor:
            futures = [executor.submit(work, i) for i in range(len(self.worlds))]
            try:
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.error or not self.active or self.active <= self.pending.keys())
                        if self.error:
                            raise self.error
                        if not self.active:
                            break
                        requests, self.pending = self.pending, {}
                    self._execute(requests)
                    with self.condition:
                        for request in requests.values():
                            request.done.set()
                        self.condition.notify_all()
                return [f.result() for f in futures]
            except BaseException as exc:
                with self.condition:
                    self.error = self.error or exc
                    self.condition.notify_all()
                raise
