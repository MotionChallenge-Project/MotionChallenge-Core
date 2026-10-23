import unittest
from types import SimpleNamespace

import numpy as np

from motion_challenge.adapters.configured import ConfiguredONNX


class RecurrentContractTests(unittest.TestCase):
    def bare(self):
        model=ConfiguredONNX.__new__(ConfiguredONNX)
        model.cache={}
        model.env=SimpleNamespace(state_dict={},total_inference_cnt=0)
        model._bind_state=lambda *args:None
        model._bind_reference=lambda *args:None
        return model

    def test_provider_reset_takes_precedence_over_previous_network_state(self):
        model=self.bare()
        model.cache={'memory':np.ones((1,2),dtype=np.float32)}
        provider=SimpleNamespace(update=lambda state:None,
                                 compute=lambda:np.zeros((1,2),dtype=np.float32))
        model.groups={'memory':[provider]}
        np.testing.assert_array_equal(model.observations(None,None)['memory'],0)

    def test_network_state_is_available_to_next_observation(self):
        model=self.bare()
        model.input_keys=['observation'];model.output_keys=[['next','memory']]
        model.session=SimpleNamespace(
            get_inputs=lambda:[SimpleNamespace(name='obs',type='tensor(float)',shape=[1,2])],
            run=lambda *args:[np.array([[3.,4.]],dtype=np.float32)])
        model.infer({'observation':np.zeros((1,2))})
        np.testing.assert_array_equal(model.env.state_dict['memory'],[[3.,4.]])
        np.testing.assert_array_equal(model.cache['memory'],[[3.,4.]])


if __name__=='__main__':unittest.main()
