import io
import tarfile
import unittest
import numpy as np
from motion_challenge.data.prepare import decode_seed


class StreamingArchiveTests(unittest.TestCase):
    def test_nonseekable_tar_member(self):
        csv = b"Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ,test_joint_dof\n0,100,0,80,0,0,90,30\n1,110,0,80,0,0,90,40\n"
        packed = io.BytesIO()
        with tarfile.open(fileobj=packed,mode="w:gz") as archive:
            info = tarfile.TarInfo("g1/csv/000000/test.csv")
            info.size = len(csv)
            archive.addfile(info,io.BytesIO(csv))
        packed.seek(0)
        with tarfile.open(fileobj=packed,mode="r|gz") as archive:
            member = next(iter(archive))
            values = decode_seed(archive.extractfile(member),["test_joint"])
        np.testing.assert_allclose(values[:,:3],[[1,0,.8],[1.1,0,.8]],atol=1e-7)
        np.testing.assert_allclose(values[:,7],[np.pi/6,2*np.pi/9],atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(values[:,3:7],axis=1),1,atol=1e-7)

class ParallelArchiveTests(unittest.TestCase):
    def test_serial_parallel_identical(self):
        import tempfile
        from pathlib import Path
        from motion_challenge.data.prepare import iter_seed_motions
        csv = b"Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ,test_joint_dof\n0,100,0,80,0,0,90,30\n1,110,0,80,0,0,90,40\n"
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"motions.tar.gz"
            with tarfile.open(path,"w:gz") as archive:
                for i in range(12):
                    info=tarfile.TarInfo(f"g1/{i}.csv");info.size=len(csv)
                    archive.addfile(info,io.BytesIO(csv))
            selected={f"g1/{i}.csv" for i in range(10)}
            serial=list(iter_seed_motions(path,selected,["test_joint"],1))
            parallel=list(iter_seed_motions(path,selected,["test_joint"],2))
            self.assertEqual([x[0] for x in serial],[x[0] for x in parallel])
            for (_,a),(_,b) in zip(serial,parallel):
                np.testing.assert_array_equal(a,b)
