
from dxl.core.debug import enter_debug
from dxl.learn.core import Session, SubgraphMakerTable
from srf.graph.reconstruction.attenuation import \
    Attenuation
from srf.model import ProjectionOrdinary
from srf.physics import CompleteLoRsModel
import numpy as np
from srf.data.listmode import ListModeData
import sys
from srf.test import TestCase
from doufo.tensor.tensor import all_close


"""
We may get u map from simulation or CT images.
So in map2umap.py from preprocess part, we have define functions:cal_umap_from_simu and cal_umap_from_ct
"""

class TestAttenuation(TestCase):
    def load_config(self):
        config = {
            'input':{
                "listmode": {
                "path_file": "/mnt/gluster/Techpi/attenuation/test2/input.npy"
                }
            },
            "algorithm": {
                "projection_model": {
                    "siddon": {
                    }
                },
                "correction": {
                    "atten_correction": {
                        "path_file": "/mnt/gluster/Techpi/attenuation/test2/u_map.npy",
                        'center': [0.0, 0.0, 0.0],
                        'size': [192.0, 192.0, 180.0],
                        'grid': [64, 64, 12]
                    }
                }
            }
        }
        return config
        
    def test_uniform_cylinder(self):
        config = self.load_config()
        model = CompleteLoRsModel('model', **config['algorithm']['projection_model']['siddon'])
        projection = ProjectionOrdinary(model)
        lors = np.load(config['input']['listmode']['path_file'])
        projection_data = ListModeData(lors, np.ones([lors.shape[0]], np.float32))
        image_config = config['algorithm']['correction']['atten_correction']
        image = np.load(image_config['path_file'])
        center = image_config['center']
        size = image_config['size']
        g = Attenuation('attenuation',projection,projection_data,image,center,size)
        g.make()
        with Session() as sess:
            value = g.run(sess)
        sess.reset()
        len_lors = np.power(lors[:,3]-lors[:,0],2)+np.power(lors[:,4]-lors[:,1],2)+np.power(lors[:,5]-lors[:,2],2)
        weight = 1.0/(np.exp(-value*len_lors)+sys.float_info.min)
        previous_result = np.load('/mnt/gluster/Techpi/attenuation/test2/result.npy')[:,7]
        assert all_close(previous_result,weight)


    