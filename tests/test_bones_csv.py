import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from motion_challenge.data.bones import load_bones_motion


class BonesTests(unittest.TestCase):
    def test_named_columns_units_and_endpoint(self):
        header=["Frame","root_translateX","root_translateY","root_translateZ",
                "root_rotateX","root_rotateY","root_rotateZ","knee_dof"]
        rows=[[0,100,200,80,0,0,90,180],[1,200,200,80,0,0,90,90]]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"motion.csv"
            with p.open("w",newline="") as f:
                w=csv.writer(f);w.writerow(header[::-1]);w.writerows([r[::-1] for r in rows])
            m=load_bones_motion(p,["knee"])
            self.assertAlmostEqual(m.duration_s,1/120)
            t,q,pos,quat=m.sample_arrays(np.array([0.,1/120]))
            np.testing.assert_allclose(pos,[[1,2,.8],[2,2,.8]],atol=1e-7)
            np.testing.assert_allclose(q[:,0],[np.pi,np.pi/2],atol=2e-7)
            np.testing.assert_allclose(quat[0],[0,0,2**-.5,2**-.5],atol=1e-7)
            rows[1][0]=3
            with p.open("w",newline="") as f:
                w=csv.writer(f);w.writerow(header);w.writerows(rows)
            with self.assertRaisesRegex(ValueError,"noncontiguous"):
                load_bones_motion(p,["knee"])


if __name__=="__main__":unittest.main()
