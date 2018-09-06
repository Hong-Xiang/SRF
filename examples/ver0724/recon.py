from dxl.core.debug import enter_debug
from dxl.learn.session import Session
from srf.data.loader import MasterLoader, CompleteWorkerLoader
from srf.graph.reconstruction.local_reconstruction import \
    LocalReconstructionGraph
from srf.graph.reconstruction.master import MasterGraph
from srf.graph.reconstruction.worker import WorkerGraph
from srf.model._backprojection import BackProjectionOrdinary
from srf.model._projection import ProjectionOrdinary
from srf.model.recon_step import ReconStep, mlem_update
from srf.physics import CompleteLoRsModel
from srf.utils.config import config_with_name

enter_debug()

RESOURCE_ROOT = '/mnt/gluster/CustomerTests/SRF/reconstruction/run0810/'


def main(config):
    model = CompleteLoRsModel('model', **config['projection_model'])
    master_loader = MasterLoader(config['shape'], config['center'], config['size'])
    worker_loader = CompleteWorkerLoader(RESOURCE_ROOT + 'mct_lors_debug.npy',
                                         RESOURCE_ROOT + 'summap.npy',
                                         config['center'],
                                         config['size'])
    recon_step = ReconStep('worker/recon',
                           ProjectionOrdinary(model),
                           BackProjectionOrdinary(model),
                           mlem_update)
    g = LocalReconstructionGraph('reconstruction',
                                 MasterGraph(
                                     'master', loader=master_loader, nb_workers=1),
                                 WorkerGraph('worker',
                                             recon_step,
                                             loader=worker_loader,
                                             task_index=0),
                                 nb_iteration=10)
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
        'size': [262.4, 262.4, 225.5],
        'shape': [128, 128, 104]
    }
    main(config)
