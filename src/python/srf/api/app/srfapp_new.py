import numpy as np
from dxl.learn.session import Session
from tqdm import tqdm

from srf.data import (ScannerClass, MasterLoader, CompleteWorkerLoader, SplitWorkerLoader,
                      ListModeDataWithoutTOF, ListModeDataSplitWithoutTOF)
from srf.data import siddonSinogramLoader
from srf.function import make_scanner, create_listmode_data
from srf.graph.reconstruction import MasterGraph, WorkerGraph
from srf.graph.reconstruction import RingEfficiencyMap, LocalReconstructionGraph
from srf.model import BackProjectionOrdinary, ProjectionOrdinary, mlem_update, ReconStep
from srf.physics import CompleteLoRsModel, SplitLoRsModel, CompleteSinoModel
# from dxl.learn.core.config import dlcc
# from dxl.learn.distribute import make_distribution_session
from srf.preprocess.function.on_tor_lors import map_process
from srf.preprocess.merge_map import merge_effmap, merge_effmap_full


class SRFApp():
    """Scalable reconstrution framework high-level API.

    A SRFApp manages the complete computing process of a SRF task.
    The task may be an MapTask, ReconTask or other tyoe of tasks.
    A SRFApp object owns two important attributes: Scanner and Task.
    SRFApp read the configure file from external client and construct 
    the two object. SRFApp also decides which kind of Scanner and Task
    it will construct from the json file.

    Attributes:

        Scanner: descripts the geometry of the medical imaging detector and
        provides some methods like return the LORs for computing efficiency
        map or locate the virtual position of simulated lors data.

        Task: descripts the computing task of this task application.

    """

    def __init__(self, job, task_index, task_config, task_name, distribution=None):
        """ 
        initialize the application 
        give the map task or recon task
        """
        # set the global configure of this application.
        # dlcc.set_global_config(task_config)
        self._scanner = self._make_scanner(task_config)
        if task_name == 'map':
            self._task = self._make_map_task(task_index, task_config)
        elif task_name == 'recon':
            self._task = self._make_recon_task(task_index, task_config)
        elif task_name == 'both':
            self._task = self._make_map_task(task_index, task_config)
            self._task = self._make_recon_task(task_index, task_config)

    def _make_scanner(self, task_config):
        """Create a specific scanner object.
        A specific scanner is built according to the user's configuration on scanner.
        The scanner will be then used in the task of this application.


        Args:
            task_config: the configuration file of this task.

        Returns:
            A scanner object.

        config = {
        "modality": "PET",
        "name": "mCT",
        "ring": {
            "inner_radius": 424.5,
            "outer_radius": 444.5,
            "axial_length": 220.0,
            "nb_rings": 104,
            "nb_blocks_per_ring": 48,
            "gap": 0.0
        },
        "block": {
            "grid": [1, 13, 1],
            "size": [20.0, 53.3, 2.05],
            "interval": [0.0, 0.0, 0.0]
        },
        "tof": {
            "resolution": 530,
            "bin": 40
            }
        }
        """
        config = task_config['scanner']['petscanner']
        if ("ring" in config):
            scanner_class = ScannerClass.CylinderPET
        return make_scanner(scanner_class, config)

    def _make_recon_task(self, task_index, task_config, distribution_config=None):
        """ Create a specific task object.
        A specific task object is built according to the user configuration on task.

        Args:
            scanner: the scanner used in this task.
            job: role of this host (master/worker).
            task_index: the index of this task.
            task_config: the configuation file of this task.
            distribution_config: hosts used to run this task.

        Returns:
            A task object.

        """
        al_config = task_config['algorithm']['projection_model']
        im_config = task_config['output']['image']
        pj_config = task_config['scanner']['petscanner']
        tof_res = self._scanner.tof_res
        tof_bin = self._scanner.tof_bin
        GAUSSIAN_FACTOR = 2.35482005
        limit = tof_res*0.15/GAUSSIAN_FACTOR*3
        tof_sigma2 = (limit**2)/9
        tof_bin = tof_bin*0.15
        if ('siddon' in al_config):
            model = CompleteLoRsModel(
                'model', tof_bin=tof_bin, tof_sigma2=tof_sigma2)
            worker_loader = CompleteWorkerLoader(task_config['input']['listmode']['path_file'],
                                                 "./summap.npy",
                                                 self._scanner,
                                                 im_config)
        elif ('siddon_sino' in al_config):
            model = CompleteSinoModel('model', pj_config)
            worker_loader = siddonSinogramLoader(pj_config,
                                                 task_config['input']['listmode']['path_file'],
                                                 "./summap.npy",
                                                 im_config)
        else:
            model = SplitLoRsModel(
                al_config['tor']['kernel_width'], tof_bin=tof_bin, tof_sigma2=tof_sigma2)
            worker_loader = SplitWorkerLoader(task_config['input']['listmode']['path_file'],
                                              "./summap.npy",
                                              self._scanner,
                                              im_config)
        master_loader = MasterLoader(self._scanner, im_config)
        recon_step = ReconStep('worker/recon',
                               ProjectionOrdinary(model),
                               BackProjectionOrdinary(model),
                               mlem_update)
        if ('mlem' in task_config['algorithm']['recon']):
            nb_iteration = task_config['algorithm']['recon']['mlem']['nb_iterations']
        else:
            nb_iteration = task_config['algorithm']['recon']['osem']['nb_iterations']
        output_filename = task_config['output']['image']['path_dataset_prefix']
        g = LocalReconstructionGraph('reconstruction',
                                     MasterGraph(
                                         'master', loader=master_loader, nb_workers=1),
                                     WorkerGraph('worker',
                                                 recon_step,
                                                 loader=worker_loader,
                                                 task_index=task_index),
                                     output_filename=output_filename,
                                     nb_iteration=nb_iteration)
        g.make()
        with Session() as sess:
            g.run(sess)

    def _make_map_task(self, task_index, task_config):
        grid, center, size, model, listmodedata = get_config(task_config)
        if task_config['output']['image']['grid'][2] == self._scanner.nb_rings:

            r1 = self._scanner.rings[0]
            grid, center, size, model, listmodedata = get_config(task_config)
            self._make_map_single_ring(r1, grid, center, size, model, listmodedata)
            merge_effmap(0, self._scanner.nb_rings, self._scanner.nb_rings, 1, './')
        else:
            for ir1 in tqdm(range(self._scanner.nb_rings)):
                r1 = self._scanner.rings[ir1]
                for ir2 in range(self._scanner.nb_rings):
                    r2 = self._scanner.rings[ir2]
                    lors = self._scanner.make_ring_pairs_lors(r1, r2)
                    projection_data = create_listmode_data[ListModeDataWithoutTOF](lors)
                    result = _compute(projection_data, grid, center, size, model)
                    # coo = psfT(grid)
                    # result = (coo * result.ravel()).reshape(grid)
                    np.save(f'./effmap/effmap_{ir1}_{ir2}.npy', result)

            merge_effmap_full(self._scanner.nb_rings, 1, './')

    def _make_map_single_ring(self, r1, grid, center, size, model, listmodedata):
        for ir in tqdm(range(0, self._scanner.nb_rings)):
            r2 = self._scanner.rings[ir]
            lors = self._scanner.make_ring_pairs_lors(r1, r2)
            if isinstance(model, SplitLoRsModel):
                lors = map_process(lors)
                projection_data = create_listmode_data[ListModeDataSplitWithoutTOF](
                    lors)
            # lors = self._scanner.make_ring_pairs_lors(r1, r2)
            else:
                projection_data = create_listmode_data[ListModeDataWithoutTOF](
                    lors)

            result = _compute(projection_data, grid, center, size, model)

            np.save('effmap_{}.npy'.format(ir), result)


def get_config(task_config):
    im_config = task_config['output']['image']
    grid = im_config['grid']
    center = im_config['center']
    size = im_config['size']
    al_config = task_config['algorithm']['projection_model']
    model, listmodedata = _get_model(al_config)
    return grid, center, size, model, listmodedata


def _get_model(config):
    if ('siddon' in config):
        model = CompleteLoRsModel('map_model')
        listmodedata = ListModeDataWithoutTOF
        kernal_width = None
    elif ('siddon_sino' in config):
        model = CompleteLoRsModel('map_model')
        listmodedata = ListModeDataWithoutTOF
        kernal_width = None
    else:
        kernal_width = config['tor']['kernel_width']
        model = SplitLoRsModel(kernal_width, 'map_model')
        listmodedata = ListModeDataSplitWithoutTOF

    return model, listmodedata


def _compute(lors, grid, center, size, model):
    map_model = BackProjectionOrdinary(model)
    t = RingEfficiencyMap('effmap', map_model, lors,
                          grid=grid, center=center, size=size)
    t.make()
    with Session() as sess:
        result = t.run()
    sess.reset()
    return result
