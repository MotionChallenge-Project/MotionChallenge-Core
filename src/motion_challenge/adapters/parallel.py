"""Bounded concurrency for independent scalar controllers, without IPC."""
from concurrent.futures import ThreadPoolExecutor, wait

from motion_challenge.simulation.vector_physics import cpu_count


class ParallelControllers:
    """Adapter opt-in: one history/RNG per slot, shared read-only model sessions."""

    def __init__(self, root, runtime, factory):
        workers = runtime.get('inference_workers', 8)
        if type(workers) is not int or workers < 1:
            raise ValueError('inference_workers must be a positive integer')
        self.workers = min(workers, cpu_count())
        self.root, self.runtime, self.factory = root, runtime, factory
        self.policies = {}
        self.pool = None
        self.closed = False

    def _check(self, slots, states, references):
        if self.closed:
            raise RuntimeError('controller is closed')
        if len(set(slots)) != len(slots) or not (len(slots) == len(states) == len(references)):
            raise ValueError('invalid parallel batch')

    def reset_batch(self, slots, states, references, seeds):
        self._check(slots, states, references)
        if len(seeds) != len(slots):
            raise ValueError('invalid seed count')
        # Model loading and reset stay serial: no global import/cache/RNG races.
        for slot, state, reference, seed in zip(slots, states, references, seeds):
            if slot not in self.policies:
                self.policies[slot] = self.factory(self.root, self.runtime)
            self.policies[slot].reset(state, reference, seed)

    def act_batch(self, slots, states, references):
        self._check(slots, states, references)
        if not set(slots) <= self.policies.keys():
            raise ValueError('act before reset')
        if self.workers == 1 or len(slots) < 2:
            return tuple(self.policies[i].act(s, r) for i, s, r in zip(slots, states, references))
        if self.pool is None:
            self.pool = ThreadPoolExecutor(self.workers, thread_name_prefix='policy')
        futures = [self.pool.submit(self.policies[i].act, s, r)
                   for i, s, r in zip(slots, states, references)]
        # Drain every call even on error; no slot may mutate after this boundary.
        wait(futures)
        return tuple(f.result() for f in futures)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        try:
            for policy in self.policies.values():
                policy.close()
        finally:
            self.policies.clear()
