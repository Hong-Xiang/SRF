from dxl.core.debug import enter_debug
from dxl.learn.core import Session, SubgraphMakerTable
from dxl.learn.core.config import update_config
from srf.data.loader import MasterLoader, WorkerLoader
from srf.graph.reconstruction.local_reconstruction import \
    LocalReconstructionGraph
from srf.graph.reconstruction.master import MasterGraph
from srf.graph.reconstruction.worker import WorkerGraph
from srf.model.backprojection import BackProjectionToR
from srf.model.projection import ProjectionToR
from srf.model.recon_step import ReconStep
from srf.physics.tor import ToRModel


enter_debug()

RESOURCE_ROOT = '/mnt/gluster/CustomerTests/SRF/reconstruction/run0605/'


def main(config):
    model = ToRModel('model', **config['projection_model'])
    master_loader = MasterLoader(config['shape'])
    worker_loader = WorkerLoader(RESOURCE_ROOT + 'lors_debug.npz',
                                 RESOURCE_ROOT + 'summap.npy')
    recon_step = ReconStep('worker/recon',
                           projection=ProjectionToR(model),
                           backprojection=BackProjectionToR(model))
    g = LocalReconstructionGraph('reconstruction',
                                 MasterGraph(
                                     'master', loader=master_loader, nb_workers=1),
                                 WorkerGraph('worker',
                                             loader=worker_loader,
                                             recon_step=recon_step,
                                             center=config['center'],
                                             size=config['size'],
                                             task_index=0))
    g.make()
    with Session() as sess:
        g.run(sess)


if __name__ == "__main__":
    config = {
        'projection_model': {
            'tof_sigma2': 162.30,
            'tof_bin': 6.0,
            'kernel_width': 3.86,
        },
        'center': [0.0, 0.0, 0.0],
        'size': [666.9, 666.9, 1422.72],
        'shape': [195, 195, 416]
    }
    main(config)
