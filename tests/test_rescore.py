import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from motion_challenge.evaluation.rescore import rescore, unchanged_rollout_rules
from motion_challenge.evaluation.verify import _score_clip, verify
from motion_challenge.resources import lafan_paths
from motion_challenge.validation import sha256_file


class RescoreTests(unittest.TestCase):
    def test_changed_failures_without_policy_or_physics(self):
        paths=lafan_paths('none');new=json.loads(paths['protocol'].read_text())
        old=copy.deepcopy(new);old['id']='test-original';old['failure']['anchor_height_m']=.01
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'original';trace_root=run/'rollout';trace_root.mkdir(parents=True)
            write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')
            csv=root/'clip.csv';csv.write_text((','.join(map(str,[0,0,.8,0,0,0,1]+[0]*29))+'\n')*4)
            manifest=json.loads(paths['suite'].read_text());manifest['files']=[{'path':'clip.csv','sha256':sha256_file(csv)}]
            suite=root/'suite.json';write(suite,manifest);old_path=root/'old.json';write(old_path,old)
            runtime=trace_root/'runtime.json';write(runtime,{'state_access':new['state_access']})
            trace=trace_root/'poses.jsonl.gz'
            with gzip.open(trace,'wt') as f:
                for i,z in enumerate([.8,.78,.76,.5,.5],1):
                    f.write(json.dumps(dict(time_s=i*.02,joint_position_rad=[0.]*29,root_position_world_m=[0.,0.,z],root_orientation_xyzw=[0.,0.,0.,1.]))+'\n')
            clip=dict(id='clip.csv',duration_s=.1,observed_time_s=.1,sample_count=5,truncated_smoke=False,trace={'path':trace.name,'sha256':sha256_file(trace)})
            task=(clip,manifest['files'][0],manifest,paths['robot'],old,root,trace_root,True)
            scored=_score_clip(task)['full'];self.assertAlmostEqual(scored['failure_time_s'],.04)
            full=dict(status='completed',full_suite=True,traces_recorded=True,termination_mode='no_termination',mujoco_version='3.17.0',backend='batch',controller_id='test',state_access=new['state_access'],suite_sha256=sha256_file(suite),robot_profile_sha256=sha256_file(paths['robot']),protocol_sha256=sha256_file(old_path),controller_config_sha256=sha256_file(runtime),clips=[scored])
            source={'adapter_commit':'original-adapter','core_commit':'original-core','adapter_dirty':False,'core_dirty':False}
            write(trace_root/'result.json',full);write(run/'result.json',{'status':'trace_verified','source':source,'checkpoint_sha256':'same-checkpoint'})
            write(trace_root/'receipt.json',{'status':'trace_verified','source':source,'source_unchanged_during_run':True})
            original_bytes=(run/'result.json').read_bytes()
            with patch('mujoco.mj_step',side_effect=AssertionError('no physics rollout permitted')):
                result=rescore(run,root/'rescored',suite,paths['robot'],old_path,paths['protocol'],lafan_paths('early')['protocol'],root)
            self.assertEqual(result['source'],source)
            self.assertNotEqual(result['scoring_source']['core_commit'],source['core_commit'])
            self.assertAlmostEqual(result['clips'][0]['failure_time_s'],.08)
            self.assertAlmostEqual(result['macro_completion'],.6)
            self.assertEqual(result['clips'][0]['sample_count'],{'until_first_failure':4,'full_motion':5})
            self.assertEqual((run/'result.json').read_bytes(),original_bytes)
            self.assertEqual(result['derivation']['original_result_sha256'],sha256_file(run/'result.json'))
            output=root/'rescored/rollout'
            checked=verify(output/'result.json',suite,paths['robot'],paths['protocol'],root,output/'runtime.json')
            self.assertEqual(checked['samples'],5)
            with self.assertRaisesRegex(ValueError,'new or empty'):
                rescore(run,root/'rescored',suite,paths['robot'],old_path,paths['protocol'],lafan_paths('early')['protocol'],root)
            trace.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'trace hash'):
                rescore(run,root/'bad',suite,paths['robot'],old_path,paths['protocol'],lafan_paths('early')['protocol'],root)

    def test_changed_dynamics_or_early_trajectory_rejected(self):
        spec=json.loads(lafan_paths('none')['protocol'].read_text())
        for key,value in [('metric_period_s',.04),('state_access','different'),('terminate_on_failure',True)]:
            changed=copy.deepcopy(spec);changed[key]=value
            with self.assertRaises(ValueError):unchanged_rollout_rules(spec,changed)
        changed=copy.deepcopy(spec);changed['nominal']['sliding_friction']=.5
        with self.assertRaises(ValueError):unchanged_rollout_rules(spec,changed)
