"""Bounded output queue; JSON/gzip work never runs in the control loop."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import threading
import time


class ArrayTraceWriter:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='trace-writer')
        self.capacity = threading.Semaphore(16)
        self.files = {}
        self.error = None
        self.timings = dict(samples=0,json_encoding_s=0.,compression_write_s=0.,buffer_submit_s=0.)

    def submit(self, fn, *args):
        self.capacity.acquire()
        def work():
            try:
                if self.error is not None: raise self.error
                return fn(*args)
            except BaseException as error:
                self.error = error
                raise
            finally: self.capacity.release()
        return self.pool.submit(work)

    def open(self, slot, path):
        path.write_bytes(b'')
        self.files[slot] = [path,[],0]

    def _flush(self, slot):
        item = self.files[slot]
        if item[1]:
            tick = time.perf_counter()
            with item[0].open('ab') as f:
                f.write(gzip.compress(''.join(item[1]).encode(),compresslevel=1,mtime=0))
            item[1],item[2] = [],0
            self.timings['compression_write_s'] += time.perf_counter()-tick

    def _write(self, ids, times, joints, positions, rotations):
        tick = time.perf_counter()
        compression = self.timings['compression_write_s']
        rows = zip(ids,times.tolist(),joints.tolist(),positions.tolist(),rotations.tolist())
        for slot,t,q,p,r in rows:
            line = json.dumps(dict(time_s=t,joint_position_rad=q,root_position_world_m=p,
                                   root_orientation_xyzw=r),allow_nan=False)+'\n'
            item = self.files[slot];item[1].append(line);item[2] += len(line)
            if item[2]>=16384:self._flush(slot)
        self.timings['json_encoding_s'] += time.perf_counter()-tick-(self.timings['compression_write_s']-compression)
        self.timings['samples'] += len(ids)

    def write(self, ids, times, joints, positions, rotations):
        tick = time.perf_counter()
        self.submit(self._write,tuple(map(int,ids)),times.copy(),joints.copy(),positions.copy(),rotations.copy())
        self.timings['buffer_submit_s'] += time.perf_counter()-tick

    def finish(self, ids):
        def flush():
            for slot in ids:
                self._flush(slot)
                del self.files[slot]
        self.submit(flush).result()

    def close(self):
        self.pool.shutdown(wait=True)
        if self.error is not None:raise self.error
