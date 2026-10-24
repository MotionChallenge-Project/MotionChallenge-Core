import gzip
import math
from pathlib import Path
import random
import tempfile
import unittest
from motion_challenge.evaluation.accumulator import MetricAccumulator
from motion_challenge.evaluation.traces import TraceWriter

class ResourceTests(unittest.TestCase):
    def test_streamed_fsum_and_memory_do_not_scale_with_samples(self):
        rng=random.Random(7)
        values=[math.ldexp(rng.random(),rng.randint(-50,50)) for _ in range(100000)]
        acc=MetricAccumulator()
        for v in values:acc.add({'a':v,'b':1.})
        self.assertEqual(acc.means(),{'a':math.fsum(values)/len(values),'b':1.})
        self.assertLess(len(acc.partials['a']),100)
        with self.assertRaisesRegex(ValueError,'keys'):acc.add({'different':2.})
        with self.assertRaisesRegex(ValueError,'non-finite'):acc.add({'a':float('nan'),'b':1.})

    def test_many_traces_bounded_buffers_and_ordered_gzip_members(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);writer=TraceWriter()
            traces=[writer.open(root/str(i)) for i in range(1100)]
            for tick in range(24):
                for trace in traces:
                    trace.write(str(tick)+'x'*1024+'\n')
                    self.assertLess(trace.size,16384)
            for trace in traces:trace.close()
            writer.close()
            expected=''.join(str(t)+'x'*1024+'\n' for t in range(24))
            for path in (root/'0',root/'1099'):
                with gzip.open(path,'rt') as f:self.assertEqual(f.read(),expected)

    def test_writer_errors_are_not_silent(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace';writer=TraceWriter();trace=writer.open(path)
            path.unlink();path.mkdir()
            trace.write('x')
            with self.assertRaises(OSError):trace.close()
            with self.assertRaises(OSError):writer.close()

class SessionTests(unittest.TestCase):
    def test_only_immutable_graph_is_shared_and_still_hash_checked(self):
        from unittest.mock import patch
        from motion_challenge.adapters.onnx import cpu_session
        from motion_challenge.validation import sha256_file
        class Session: pass
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model';path.write_bytes(b'graph')
            with patch('motion_challenge.adapters.onnx.ort.InferenceSession',side_effect=lambda *a,**k:Session()) as factory:
                a=cpu_session(path,sha256_file(path));b=cpu_session(path,sha256_file(path))
                self.assertIs(a,b)
                self.assertEqual(factory.call_count,1)
                with self.assertRaisesRegex(ValueError,'checksum'):
                    cpu_session(path,'0'*64)
