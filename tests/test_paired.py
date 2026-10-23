import copy
import unittest
from motion_challenge.paired import combine

def fixture():
    clip={'id':'x','duration_s':2.,'success':False,'completion':.5,'failure_time_s':1.,'failure_reasons':['height'],'sample_count':50,'metric_means_over_observed_prefix':{'mpbpe':.1}}
    return dict(controller_id='x',controller_config_sha256='a',suite_sha256='b',robot_profile_sha256='c',mujoco_version='3.17.0',backend='batch',clips=[clip],macro_metrics_over_observed_prefix={'mpbpe':.1})

class PairedResultTests(unittest.TestCase):
    def test_rejects_different_config_or_first_failure(self):
        a=fixture()
        for mutate in (lambda b:b.update(controller_config_sha256='wrong'),lambda b:b['clips'][0].update(failure_time_s=1.02),lambda b:b['clips'][0].update(success=True),lambda b:b['clips'].append(copy.deepcopy(b['clips'][0]))):
            b=copy.deepcopy(a);mutate(b)
            with self.assertRaises(ValueError):combine(a,b,{},'weights')
    def test_one_outcome_with_two_metric_scopes(self):
        a=fixture();b=copy.deepcopy(a);b['macro_metrics_over_observed_prefix']={'mpbpe':10.}
        result=combine(a,b,{'adapter_commit':'same'},'weights')
        self.assertEqual(result['metrics']['until_first_failure']['mpbpe'],.1)
        self.assertEqual(result['metrics']['full_motion']['mpbpe'],10.)
        self.assertEqual(result['source']['adapter_commit'],'same')
