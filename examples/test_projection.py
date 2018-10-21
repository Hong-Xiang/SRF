from dxl.core.debug import enter_debug
from dxl.learn.session import Session
from srf.data import MasterLoader, siddonProjectionLoader, siddonImageLoader
from srf.graph.reconstruction import LocalProjectionGraph,MasterGraph,WorkerGraph
from srf.model import BackProjectionOrdinary,ProjectionOrdinary, BprojStep, ProjStep
from srf.physics import CompleteLoRsModel
from srf.utils.config import config_with_name
import numpy as np
enter_debug()

RESOURCE_ROOT = '/mnt/gluster/Techpi/brain16/recon/data/'


def main(config):
    model = CompleteLoRsModel('model', **config['projection_model'])
    master_loader = MasterLoader(config['shape'], config['center'], config['size'])
    worker_loader = siddonImageLoader(RESOURCE_ROOT + 'd12.h5',
                                         config['center'],
                                         config['size'],
                                         config['shape'])
    proj_step = ProjStep('worker/projection',
                           ProjectionOrdinary(model))
    g = LocalProjectionGraph('projection',
                                 MasterGraph(
                                     'master', loader=master_loader, nb_workers=1),
                                 WorkerGraph('worker',
                                             proj_step,
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
        'size': [220.0, 220.0, 33.4],
        'shape': [110, 110, 10]
    }
    main(config)
