from dxl.learn.core import DistributeTask, Barrier, make_distribute_session
from dxl.learn.core import Master, Barrier, ThisHost, ThisSession, Tensor

# from ..graph import MasterGraph
# from ..graph import WorkerGraph
import json


class SRFTask(DistributeTask):
    class KEYS(DistributeTask.KEYS):
        class STEPS(DistributeTask.KEYS.STEPS):
            pass

    def __init__(self, job, task_index, task_info, distribute_configs):
        super().__init__(distribute_configs)
        self.job = job
        self.task_index = task_index
        self.task_info = task_info
        # initialize the cluster infomation
        self.cluster_init(job, task_index)
        # parse the task informations
        self.parse_task()

        self.pre_works()
        # do pre works needed before create the maps

        # create the master and worker graphs
        self.make_graphs()

        # binding static local data
        self.bind_local_data()

        # set the steps
        self.make_steps()
        # create the distribute session.
        make_distribute_session()

    def parse_task(self):
        self.task_type = self.task_info['task_type']
        self.work_directory = self.task_info['work_directory']

    def pre_works(self):
        """
        do some work like data preprocessing.
        """
        pass

    def make_graphs(self):
        self.create_master_graph()
        self.create_worker_graphs()

    def create_master_graph(self):
        pass

    def create_worker_graphs(self):
        pass

    def bind_local_data(self):
        """
        bind static data into the graph.
        """
        pass

    def make_steps(self):
        """
        the
        """
        pass


# class TaskBase:
#     def __init__(self, task_info):
#         self._info = task_info

# class TaskA(TaskBase):
#     pass

# class TaskB(TaskBase):
#     pass

# class TaskInfoBase:
#     task_cls = None

# class TaskAInfo(TaskInfoBase):
#     task_cls = TaskA

# class TaskBInfo(TaskInfoBase):
#     task_cls = TaskB

# class TaskFactory:
#     @classmethod
#     def make_task(cls, t: TaskInfoBase):
#         return t.task_cls(t)


# def test():
#     info = TaskAInfo()
#     task = TaskFactory.make_task(info)
#     task = TaskFactory.make_task(info)
