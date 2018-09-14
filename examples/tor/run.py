from dxl.core.debug import enter_debug
from dxl.learn import Session
from srf.data.loader import MasterLoader, SplitWorkerLoader
from srf.graph.reconstruction.local_reconstruction import \
    LocalReconstructionGraph
from srf.graph.reconstruction.master import MasterGraph
from srf.graph.reconstruction.worker import WorkerGraph
from srf.model import BackProjectionOrdinary, ProjectionOrdinary
from srf.model.recon_step import ReconStep, mlem_update_normal, mlem_update
from srf.physics import SplitLoRsModel

enter_debug()

RESOURCE_ROOT = '/mnt/gluster/CustomerTests/SRF/reconstruction/run0710/'


def main(config):
    model = SplitLoRsModel(**config['projection_model'])
    master_loader = MasterLoader(config['shape'], config['center'], config['size'])
    worker_loader = SplitWorkerLoader(RESOURCE_ROOT + 'lors_debug.npz',
                                      RESOURCE_ROOT + 'summap.npy',
                                      config['center'],
                                      config['size'])
    recon_step = ReconStep('worker/recon',
                           ProjectionOrdinary(model),
                           BackProjectionOrdinary(model),
                           mlem_update_normal)
    g = LocalReconstructionGraph('reconstruction',
                                 MasterGraph(
                                     'master', loader=master_loader, nb_workers=1),
                                 WorkerGraph('worker',
                                             recon_step,
                                             loader=worker_loader,
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
        'size': [666.9, 666.9, 1419.3],
        'shape': [195, 195, 415]
    }
    main(config)
