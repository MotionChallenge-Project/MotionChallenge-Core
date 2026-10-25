import threading
import unittest
from unittest.mock import patch
import numpy as np
from motion_challenge.adapters.parallel import ParallelControllers

class Stateful:
    def __init__(self, root, runtime):
        self.closed = False
    def reset(self, state, reference, seed):
        self.rng = np.random.default_rng(seed)
        self.total = 0.
    def act(self, state, reference):
        self.total += self.rng.normal() + state
        return self.total
    def close(self):
        self.closed = True

class ParallelAdapterTests(unittest.TestCase):
    def test_reordering_partial_reset_and_per_slot_rng(self):
        with patch('motion_challenge.adapters.parallel.cpu_count', return_value=4):
            a = ParallelControllers(None, {'inference_workers':1}, Stateful)
            b = ParallelControllers(None, {'inference_workers':8}, Stateful)
        self.assertEqual(b.workers,4)
        for obj in (a,b): obj.reset_batch([8,2,19],[0]*3,[None]*3,[42,43,44])
        try:
            for step,slots in enumerate(([19,8,2],[8],[2,19],[8,19,2])):
                states=[step+.1*i for i in slots]
                self.assertEqual(a.act_batch(slots,states,[None]*len(slots)),b.act_batch(slots,states,[None]*len(slots)))
                if step==1:
                    for obj in (a,b):obj.reset_batch([19],[0],[None],[45])
            with self.assertRaises(ValueError): b.act_batch([8,8],[0,0],[None,None])
            with self.assertRaises(ValueError): b.act_batch([50],[0],[None])
        finally:
            policies=list(b.policies.values());a.close();b.close();b.close()
        self.assertTrue(all(p.closed for p in policies))
        with self.assertRaises(RuntimeError):b.act_batch([8],[0],[None])

    def test_failure_drains_outstanding_calls(self):
        barrier=threading.Barrier(2);finished=threading.Event()
        class Failing(Stateful):
            def act(self,state,ref):
                barrier.wait(timeout=5)
                if state==0:raise RuntimeError('model failure')
                finished.set();return 1
        with patch('motion_challenge.adapters.parallel.cpu_count',return_value=2):
            obj=ParallelControllers(None,{'inference_workers':2},Failing)
        obj.reset_batch([0,1],[0,0],[None,None],[1,2])
        try:
            with self.assertRaisesRegex(RuntimeError,'model failure'):obj.act_batch([0,1],[0,1],[None,None])
            self.assertTrue(finished.is_set())
        finally:obj.close()

    def test_invalid_worker_limit(self):
        for value in (0,-1,True,1.5,'8'):
            with self.assertRaises(ValueError):ParallelControllers(None,{'inference_workers':value},Stateful)
