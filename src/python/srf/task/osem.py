import json, h5py, time, logging
import numpy as np

from tqdm import tqdm

from dxl.learn.core import Barrier, make_distribute_session
from dxl.learn.core import Master, Barrier, ThisHost, ThisSession, Tensor

from ..graph.master import MasterGraph
from ..graph.worker import WorkerGraphLOR
from ..services.utils import print_tensor, debug_tensor
from ..preprocess.preprocess import partition as preprocess_tor

from ..app.srfapp import logger


from .data import ImageInfo, DataInfo, MapInfo
from .srftask import SRFTask




class OSEMTask(SRFTask):
    class KEYS(SRFTask.KEYS):
        class STEPS(SRFTask.KEYS.STEPS):
            INIT = 'init_step'
            RECON = 'recon_step'
            MERGE = 'merge_step'
        class Task_Infos(SRFTask.KEYS.TASK_INFOS):
            pass
    def __init__(self, job, task_index, task_configs, distribute_configs):
        super.__init__(self, job, task_index, task_configs, distribute_configs)
        


    def pre_works(self):
        pass



    def create_master_graph(self, x):
        mg = MasterGraph(x, self.nb_workers(), self.ginfo_master())
        self.add_master_graph(mg)
        logger.info("Global graph created.")
        return mg


    def create_worker_graphs(self, image_info, data_info: DataInfo):
        # for i in range(self.nb_workers()):
        #     logger.info("Creating local graph for worker {}...".format(i))
        #     self.add_worker_graph(
        #         WorkerGraphLOR(
        #             self.master_graph,
        #             image_info,
        #             {a: data_info.lor_shape(a, i)
        #             for a in ['x', 'y', 'z']},
        #             i,
        #             self.ginfo_worker(i),
        #         ))
        logger.info("All local graph created.")
        return self.worker_graphs



    def make_steps(self):
        KS = self.KEYS.STEPS
        init_step = self._make_init_step()
        recon_step = self._make_recon_step()
        merge_step = self._make_merge_step()
        self.steps = {
            KS.INIT: init_step,
            KS.RECON: recon_step,
            KS.MERGE: merge_step,
        }
        
    
    def make_init_step(self, name='init'):
        init_barrier = Barrier(name, self.hosts, [self.master_host],
                                [[g.tensor(g.KEYS.TENSOR.INIT)]
                                for g in self.worker_graphs])
        master_op = init_barrier.barrier(self.master_host)
        worker_ops = [init_barrier.barrier(h) for h in self.hosts]
        self.add_step(name, master_op, worker_ops)
        return name

    def make_recon_step(self, name='recon'):
        recons = [[g.tensor(g.KEYS.TENSOR.UPDATE)] for g in self.worker_graphs]
        calculate_barrier = Barrier(
            name, self.hosts, [self.master_host], task_lists=recons)
        master_op = calculate_barrier.barrier(self.master_host)
        worker_ops = [calculate_barrier.barrier(h) for h in self.hosts]
        self.add_step(name, master_op, worker_ops)
        return name

    def make_merge_step(self, name='merge'):
        """
        """
        merge_op = self.master_graph.tensor(self.master_graph.KEYS.TENSOR.UPDATE)
        merge_barrier = Barrier(name, [self.master_host], self.hosts, [[merge_op]])
        master_op = merge_barrier.barrier(self.master_host)
        worker_ops = [merge_barrier.barrier(h) for h in self.hosts]
        self.add_step(name, master_op, worker_ops)
        return name


    def load_reconstruction_configs(self, config=None):
        pass

    def run(self):
        KS = self.KEYS.STEPS
        # self.run_step_of_this_host(self.steps[KS.INIT])
        # logger.info('STEP: {} done.'.format(self.steps[KS.INIT]))
        nb_steps = 10
        nb_subsets = 10
        for i in tqdm(range(nb_steps), ascii=True):
            for j in tqdm(range(nb_subsets),ascii=True):
                self.run_step_of_this_host(self.steps[KS.INIT])
                logger.info('STEP: {} done.'.format(self.steps[KS.INIT]))
                
                self.run_step_of_this_host(self.steps[KS.RECON])
                logger.info('STEP: {} done.'.format(self.steps[KS.RECON]))

                self.run_step_of_this_host(self.steps[KS.MERGE])
                logger.info('STEP: {} done.'.format(self.steps[KS.MERGE]))

                self.run_and_save_if_is_master(
                    self.master_graph.tensor('x'),
                    './debug/mem_lim_result_{}_{}.npy'.format(i, j))
        logger.info('Recon {} steps {} done.'.format(nb_steps, nb_subsets))
        # time.sleep(5)

    def bind_local_data(self, data_info, task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if ThisHost.is_master():
            logger.info("On Master node, skip bind local data.")
            return
        else:
            logger.info(
                "On Worker node, local data for worker {}.".format(task_index))
            emap, lors = self.load_local_data(data_info, task_index)
            self.worker_graphs[task_index].assign_efficiency_map(emap)
            self.worker_graphs[task_index].assign_lors(lors)
    
    def bind_local_data_splitted(self, lors):
        step = 10000
        nb_osem = 10
        for i in range(nb_osem):
            self.worker_graphs[self.task_index].tensors['osem_{}'.format(i)] = self.tensor('lorx').assign(lors[i*step: (i+1)*step, ...])
        
        # when run
        ThisSession.run()


    def run_and_save_if_is_master(self, x, path):
        if ThisHost.is_master():
            if isinstance(x, Tensor):
                x = x.data
            result = ThisSession.run(x)
            np.save(path, result)


    def load_data(self, file_name, lor_range=None):
        if file_name.endswith('.npy'):
            data = np.load(file_name)
            if lor_range is not None:
                data = data[lor_range[0]:lor_range[1], :]
        elif file_name.endswith('.h5'):
            with h5py.File(file_name, 'r') as fin:
                if lor_range is not None:
                    data = np.array(fin['data'][lor_range[0]:lor_range[1], :])
                else:
                    data = np.array(fin['data'])
        return data


    # Load datas
    def load_local_data(self, data_info: DataInfo, task_index):

        lors = {}
        for a in ['x', 'y', 'z']:
            msg = "Loading {} LORs from file: {}, with range: {}..."
            logger.info(msg.format(
                a, data_info.lor_file(a), data_info.lor_range(a)))
            lors[a] = self.load_data(data_info.lor_file(a), data_info.lor_range(a))
        logger.info('Loading local data done.')
        return lors


    def load_local_effmap(self, map_info: MapInfo, task_index):
        logger.info("Loading efficiency map from file: {}...".format(
            map_info.map_file()))
        emap = self.load_data(map_info.map_file())
        return emap


class TorTask(OSEMTask):
    class KEYS(OSEMTask.KEYS):
        class STEPS(OSEMTask.KEYS.STEPS):
            INIT = 'init_step'
            RECON = 'recon_step'
            MERGE = 'merge_step'

    def __init__(self, job, task_index, task_configs, distribute_configs):
        super.__init__(job, task_index, task_configs, distribute_configs)        


class SiddonTask(OSEMTask):
    pass
