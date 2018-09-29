import os
import numpy as np
import  pytest
from srf.test import TestCase
from doufo.tensor.tensor import all_close

class TestBBS(TestCase):
    def get_16module_derenzo_path(self):
        return '/mnt/gluster/Techpi/brain16/recon/castor/derenzo'

    def test_16module_dernezo(self):
        cmd = '. '+self.get_16module_derenzo_path()+'/run_castor.sh'
        os.system(cmd)
        fortest_result = np.fromfile(self.get_16module_derenzo_path()+'/derenzo_cas/derenzo_cas_it10.img',np.float32)
        new_recon_result = np.fromfile('./derenzo_cas/derenzo_cas_it10.img',np.float32)
        assert all_close(fortest_result,new_recon_result)

