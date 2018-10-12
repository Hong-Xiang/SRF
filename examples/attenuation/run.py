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
    grid = image_config['grid']
    center = image_config['center']
    size = image_config['size']
    g = Attenuation('reconstruction',projection,projection_data,image,grid,center,size)
    g.make()
    with Session() as sess:
        value = g.run(sess)
    sess.reset()
    weight = 1.0/(np.exp(-value)+sys.float_info.min)
    result = np.hstack((lors,weight.reshape((weight.size,1))))
    np.save('result.npy',result)


if __name__ == "__main__":
    config = {
        'input':{
            "listmode": {
            "path_file": "/home/twj2417/QT/atten/atten/ecat_hoffman/input.npy"
            }
        },
        "algorithm": {
            "projection_model": {
                "siddon": {
                }
            },
            "correction": {
                "atten_correction": {
                    "path_file": "/home/twj2417/QT/atten/atten/u_map.npy",
                    'center': [0.0, 0.0, 0.0],
                    'size': [192.0, 192.0, 180.0],
                    'grid': [64,64,12]
                }
            }
        }
    }
    main(config)