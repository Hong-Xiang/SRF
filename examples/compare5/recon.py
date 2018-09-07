from dxl.core.debug import enter_debug
from dxl.learn.session import Session
from srf.data.loader import MasterLoader, CompleteWorkerLoader
from srf.graph.reconstruction.local_reconstruction import \
    LocalReconstructionGraph
from srf.graph.reconstruction.master import MasterGraph
from srf.graph.reconstruction.worker import WorkerGraph
from srf.model._backprojection import BackProjectionOrdinary
from srf.model._projection import ProjectionOrdinary
from srf.model.recon_step import ReconStep, mlem_update_normal, mlem_update
from srf.physics import CompleteLoRsModel
from srf.utils.config import config_with_name

enter_debug()


def main(config, name):
    # map_path = "/mnt/gluster/Techpi/brain16/recon/SRF/effmap/summap.npy"
    # lor_path = f"/mnt/gluster/Techpi/brain16/simu/{name}.npy"
    map_path = "/mnt/gluster/Techpi/brain16/recon/bbs/brain16_fine.npy"
    lor_path = f"/mnt/gluster/Techpi/brain16/recon/SRF/{name}.npy"
    model = CompleteLoRsModel('model', **config['projection_model'])
    master_loader = MasterLoader(config['shape'], config['center'], config['size'])
    worker_loader = CompleteWorkerLoader(lor_path,
                                         map_path,
                                         config['center'],
                                         config['size'])
    recon_step = ReconStep('worker/recon',
                           ProjectionOrdinary(model),
                           BackProjectionOrdinary(model),
                           mlem_update_normal)
                           # mlem_update)
    g = LocalReconstructionGraph('reconstruction',
                                 MasterGraph(
                                     'master', loader=master_loader, nb_workers=1),
                                 WorkerGraph('worker',
                                             recon_step,
                                             loader=worker_loader,
                                             task_index=0),
                                 nb_iteration=20)
    g.make()
    with Session() as sess:
        g.run(sess)


import json

if __name__ == "__main__":
    config = {
        'projection_model': {
            'tof_sigma2': 50000000,
            'tof_bin': 40.0,
        },
        'center': [0.0, 0.0, 0.0],
        'size': [150.0, 150.0, 30.0],
        'shape': [100, 100, 20]
    }
    with open("name.json") as fin:
        dct = json.load(fin)
    main(config, dct['name'])
