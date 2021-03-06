import os
import numpy as np
import  pytest
from srf.test import TestCase
from doufo.tensor.tensor import all_close

class TestBBS(TestCase):
    def get_mct_path(self):
        return '/mnt/gluster/Techpi/mct/recon/SRF_tor'

    def test_16module_dernezo(self):
        cmd = '. '+self.get_mct_path()+'/run_tor.sh'
        os.system(cmd)
        fortest_result = np.load(self.get_mct_path()+'/recon_10.npy')
        new_recon_result = np.load('recon_10.npy')
        assert all_close(fortest_result,new_recon_result)

