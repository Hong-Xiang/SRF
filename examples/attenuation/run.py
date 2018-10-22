from dxl.core.debug import enter_debug
from dxl.learn.core import Session, SubgraphMakerTable
from srf.graph.reconstruction.attenuation import \
    Attenuation
from srf.model import ProjectionOrdinary
from srf.physics import CompleteLoRsModel
import numpy as np
from srf.data.listmode import ListModeData
import sys

enter_debug()


def main(config):
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
    result = np.hstack((lors,weight.reshape((weight.size,1))))
    np.save('result.npy',result)


if __name__ == "__main__":
    config = {
        'input':{
            "listmode": {
            "path_file": "/home/twj2417/QT/atten/atten/test1/input.npy"
            }
        },
        "algorithm": {
            "projection_model": {
                "siddon": {
                }
            },
            "correction": {
                "atten_correction": {
                    "path_file": "/home/twj2417/SRF/SRF/examples/attenuation/u_map_cylinder.npy",
                    'center': [0.0, 0.0, 0.0],
                    'size': [400.0, 400.0, 200.0],
                    'grid': [400, 400, 1]
                }
            }
        }
    }
    main(config)