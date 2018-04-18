from dxl.learn.core import DistributeTask, Barrier, make_distribute_session
from dxl.learn.core import Master, Barrier, ThisHost, ThisSession, Tensor

from ..graph.master import MasterGraph
from ..graph.worker import WorkerGraph





class SRFTask(DistributeTask):
    class KEYS(DistributeTask.KEYS):
        class STEPS(DistributeTask.KEYS.STEPS):
            INIT = 'init_step'
            RECON = 'recon_step'
            MERGE = 'merge_step'
        class TASK_INFOS:
            pass

    def __init__(self, job, task_index, task_configs, distribute_configs):
        super.__init__(distribute_configs)
        self.job = job
        self.task_index = task_index
        self.task_info = {}
        # initialize the cluster infomation
        self.cluster_init(self.job, task_index)
        # load the task informations
        self._load_task_configs(task_configs)

        self._pre_works()
        
        # create the master and worker graphs
        self._make_graphs()

        # binding local data

        # set the steps
        self._make_steps()
        # create the distribute session.
        make_distribute_session()

    def _pre_works(self):
        """
        do some work like data preprocessing.
        """
        pass


    def _load_task_configs(self, task_configs):
        """
        """
        pass


    def _make_graphs(self):
        self._create_master_graph()
        self._create_worker_graphs()


    def _create_master_graph(self):
        pass


    def _create_worker_graphs(self):
        pass


    def _make_steps(self):
        pass
        
    
    def run(self):
        """
        the 
        """
        pass


