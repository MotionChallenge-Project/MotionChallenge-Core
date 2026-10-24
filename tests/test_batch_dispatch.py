"""Exercise the public process boundary, including reset and worker failure."""
import gzip, hashlib, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class DispatchTests(unittest.TestCase):
    def test_process_parity_and_error_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root/'adapter.py').write_text('''from motion_challenge.sdk import ControlCommand
class Policy:
    def __init__(self, fail): self.fail = fail
    def reset(self, state, reference, seed): self.count = 0
    def act(self, state, reference):
        if self.fail: raise ValueError('deliberate adapter failure')
        self.count += 1
        n = len(state.joint_names)
        return ControlCommand((self.count*.001,)*n,(0.,)*n,(20.,)*n,(1.,)*n,(0.,)*n)
    def close(self): pass
def create(root, config): return Policy(config.get('fail',False))
''')
            manifest = json.loads((ROOT/'suites/lafan/dataset.json').read_text())
            manifest['files'] = []
            for i in range(3):
                path = root/f'clip{i}.csv'
                row = ','.join(str(x) for x in ([0,0,.8,0,0,0,1]+[0]*29))+'\n'
                path.write_text(row*(i+2))
                manifest['files'].append({'path':path.name, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            (root/'suite.json').write_text(json.dumps(manifest))
            runtime = {'state_access':'ideal-proprioceptive-no-position-v1', 'id':'test-policy','adapter_root':str(root),'entrypoint':'adapter:create',
                       'control_period_s':.013,'reference_offsets_s':[0.,.013],
                       'reference_history_s':0,'reference_preview_s':.013}
            cfg = root/'runtime.json';cfg.write_text(json.dumps(runtime))
            command = [sys.executable,'-m','motion_challenge.run','--controller',str(cfg),
                       '--suite',str(root/'suite.json'),'--robot',str(ROOT/'robots/g1-29dof-cpu-v0.json'),
                       '--protocol',str(ROOT/'suites/lafan/protocol.mujoco317-ideal-proprioception-early-v1.json'),
                       '--data',str(root),'--batch-size','2']
            env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            for mode in ('serial','batch'):
                runner = ([sys.executable,str(ROOT/'tests/scalar_oracle.py'),*command[3:]] if mode=='serial' else command)
                result = subprocess.run(runner+['--output',str(root/(mode+'.json'))],
                                        env=env,capture_output=True,text=True,timeout=40)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            serial = json.loads((root/'serial.json').read_text());batch = json.loads((root/'batch.json').read_text())
            for a,b in zip(serial['clips'],batch['clips']):
                self.assertEqual(gzip.open(root/a['trace']['path'],'rt').read(),gzip.open(root/b['trace']['path'],'rt').read())
                for left,right in ((a,b),(a['metric_prefix'],b['metric_prefix'])):
                    for key in left:
                        if key in ('metric_means_over_observed_prefix','last_sample'):
                            # NumPy/SciPy may round differently; trajectory and
                            # failure boundaries remain exactly equal.
                            self.assertEqual(left[key].keys(),right[key].keys())
                            for metric in left[key]:
                                self.assertAlmostEqual(left[key][metric],right[key][metric],delta=4e-15)
                        elif key not in ('wall_time_s','trace','metric_prefix'):
                            self.assertEqual(left[key],right[key])
            runtime['fail'] = True;cfg.write_text(json.dumps(runtime))
            result = subprocess.run(command+['--backend','batch','--output',str(root/'error.json')],
                                    env=env,capture_output=True,text=True,timeout=40)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(json.loads((root/'error.json').read_text())['status'],'infrastructure_error')
