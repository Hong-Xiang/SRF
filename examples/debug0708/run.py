from dxl.core.debug import enter_debug
from dxl.learn.core import Session, SubgraphMakerTable
from dxl.learn.core.config import update_config
from srf.data.loader import MasterLoader, WorkerLoader
from srf.graph.reconstruction.local_reconstruction import \
    LocalReconstructionGraph
from srf.graph.reconstruction.master import MasterGraph
from srf.graph.reconstruction.worker import WorkerGraph
from srf.model._backprojection import BackProjectionToR
from srf.model._projection import ProjectionToR
from srf.model.recon_step import ReconStep
from srf.physics.tor import ToRModel


enter_debug()

RESOURCE_ROOT = '/home/hongxwing/Workspace/dev.SRF/debug0708/'


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


from dxl.function import fmap

if __name__ == "__main__":
    tr = 200.0
    gf, cf = 2.35482005, 0.15
    limit = tr * cf / gf * 3
    config = {
        'projection_model': {
            'tof_sigma2': limit**2 / 9.0,
            'tof_bin': 25 * cf,
            'kernel_width': 6.0,
        },
        'center': [0.0, 0.0, 0.0],
        'size': [262.4, 262.4, 213.2],
        'shape': [128, 128, 104]
    }
    main(config)
