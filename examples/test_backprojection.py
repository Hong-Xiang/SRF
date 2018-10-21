from dxl.core.debug import enter_debug
from dxl.learn.session import Session
from srf.data import MasterLoader, siddonProjectionLoader
from srf.graph.reconstruction import LocalBackprojectionGraph,MasterGraph,WorkerGraph
from srf.model import BackProjectionOrdinary,ProjectionOrdinary, BprojStep
from srf.physics import CompleteLoRsModel
from srf.utils.config import config_with_name
import numpy as np
enter_debug()

RESOURCE_ROOT = '/mnt/gluster/Techpi/brain16/recon/data/'


def main(config):
    model = CompleteLoRsModel('model', **config['projection_model'])
    master_loader = MasterLoader(config['shape'], config['center'], config['size'])
    worker_loader = siddonProjectionLoader(RESOURCE_ROOT + 'ring_nogap.h5',
                                         config['center'],
                                         config['size'])
    bproj_step = BprojStep('worker/backprojection',
                           BackProjectionOrdinary(model))
    g = LocalBackprojectionGraph('backprojection',
                                 MasterGraph(
                                     'master', loader=master_loader, nb_workers=1),
                                 WorkerGraph('worker',
                                             bproj_step,
                                             loader=worker_loader,
                                             task_index=0))
    g.make()
    with Session() as sess:
        g.run(sess)



if __name__ == "__main__":
    config = {
        'projection_model': {
            'tof_sigma2': 530,
            'tof_bin': 40.0,
        },
        'center': [0.0, 0.0, 0.0],
        'size': [238.0, 238.0, 33.4],
        'shape': [119, 119, 10]
    }
    main(config)
