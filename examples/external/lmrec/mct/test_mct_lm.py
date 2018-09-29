import os
import numpy as np
import  pytest
from srf.test import TestCase

class TestBBS(TestCase):
    def get_mct_path(self):
        return '/mnt/gluster/Techpi/mct/recon/lmrec'

    def test_16module_dernezo(self):
        cmd = '. '+self.get_mct_path()+'/run_lmrec.sh'
        os.system(cmd)
        fortest_result = np.fromfile(self.get_mct_path()+'/output_10.rec',dtype='float32')
        new_recon_result = np.fromfile('output_10.rec',dtype='float32')
        assert (fortest_result==new_recon_result).all()
