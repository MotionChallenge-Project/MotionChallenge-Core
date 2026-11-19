import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from motion_challenge.evaluation.tensor_metrics import score
from motion_challenge.evaluation.reproduction import compare,ROLLOUT_FILES

class TensorMetricTests(unittest.TestCase):
 def test_translation_and_height_are_separate(self):
  ref=torch.tensor([[[0.,0.,1.],[1.,0.,1.],[0.,1.,2.]]])
  actual=ref+torch.tensor([3.,-2.,.2])
  quat=torch.tensor([[0.,0.,0.,1.]])
  joints=torch.zeros((1,29))
  values,failed=score(actual,quat,joints,ref,quat,joints,0,[0,1,2],{'failure':{'anchor_height_m':.25,'gravity_z':.8}})
  np.testing.assert_allclose(values.numpy()[0],[np.sqrt(13.04),.2,.2,0,.2,0],atol=1e-6)
  self.assertFalse(bool(failed[0]))
 def test_yaw_and_quaternion_sign(self):
  ref=torch.tensor([[[0.,0.,1.],[1.,0.,1.],[0.,1.,2.]]],dtype=torch.float64)
  actual=ref.clone();actual[...,0]=-ref[...,1]+3;actual[...,1]=ref[...,0]-2
  a=torch.tensor([[0.,0.,np.sin(np.pi/4),np.cos(np.pi/4)]])
  r=torch.tensor([[0.,0.,0.,1.]])
  joints=torch.zeros((1,29));protocol={'failure':{'anchor_height_m':.25,'gravity_z':.8}}
  positive,failed=score(actual,a,joints,ref,r,joints,0,[0,1,2],protocol)
  negative,_=score(actual,-a,joints,ref,-r,joints,0,[0,1,2],protocol)
  self.assertLess(float(positive[0,2]),1e-12);torch.testing.assert_close(positive,negative)
  self.assertFalse(bool(failed[0]))
 def test_failure_is_strict(self):
  ref=torch.zeros((2,1,3),dtype=torch.float64);actual=ref.clone();actual[:,0,2]=torch.tensor([.25,.25001],dtype=torch.float64)
  q=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1);j=torch.zeros((2,29))
  _,failed=score(actual,q,j,ref,q,j,0,[0],{'failure':{'anchor_height_m':.25,'gravity_z':.8}})
  self.assertEqual(failed.tolist(),[False,True])

class ReproductionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  metric={'mpjpe_world_m':.2,'mpjpe_xy_aligned_m':.1,'mpjpe_xy_yaw_aligned_m':.05}
  self.result={'status':'unverified_full_rollout','traces_recorded':True,'dataset':'lafan','protocol_id':'test','protocol_sha256':'p','state_access':'strict','runtime_sha256':'r','checkpoint_sha256':'w','engine':{'worlds':40,'njmax':4096},'control_period_s':.02,'physics_dt_s':.005,'dataset_identity':{'poses_sha256':'d'},'source_sha256':{name:'source' for name in ROLLOUT_FILES},'adapter_sha256':{'tensor_adapter.py':'adapter'},'converted_weights_sha256':{'weights.safetensors':'weights','parity.json':'report'},'summary':{'success_count':1,'completion':1.,'full':metric.copy(),'prefix':metric.copy()},'clips':[{'name':'motion','samples':10,'duration_s':.2,'success':True,'completion':1.,'full':metric.copy(),'prefix':metric.copy()}]}
 def write(self,name,result):
  directory=self.root/name;directory.mkdir();p=directory/'result.json';p.write_text(json.dumps(result));digest=hashlib.sha256(p.read_bytes()).hexdigest()
  (directory/'verification.json').write_text(json.dumps({'verification':'independent_cpu_trace_recomputation_passed','result_sha256':digest,'clips':1}))
  return directory
 def test_numerical_differences_are_reported(self):
  other=copy.deepcopy(self.result);other['clips'][0]['full']['mpjpe_world_m']+=.01;other['summary']['full']['mpjpe_world_m']+=.01
  report=compare(self.write('a',self.result),self.write('b',other))
  self.assertFalse(report['numerically_identical_metrics'])
  self.assertAlmostEqual(report['max_per_motion_abs_difference']['full.mpjpe_world_m'],.01)
 def test_physics_identity_mismatch_rejected(self):
  other=copy.deepcopy(self.result);other['engine']['njmax']=1024
  with self.assertRaisesRegex(ValueError,'engine'):compare(self.write('a',self.result),self.write('b',other))
 def test_rollout_code_mismatch_rejected(self):
  other=copy.deepcopy(self.result);other['source_sha256']['simulation/warp_physics.py']='changed'
  with self.assertRaisesRegex(ValueError,'Rollout source'):compare(self.write('a',self.result),self.write('b',other))
 def test_tampered_result_receipt_rejected(self):
  a=self.write('a',self.result);b=self.write('b',self.result);(b/'result.json').write_text(json.dumps(self.result)+' ')
  with self.assertRaisesRegex(ValueError,'receipt'):compare(a,b)

