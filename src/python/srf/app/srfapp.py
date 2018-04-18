"""
Reconstruction with memory optimized:

sample code :: python

  import click
  import logging
  from dxl.learn.graph.reconstruction.reconstruction import main

  logger = logging.getLogger('dxl.learn.graph.reconstruction')
  logger.setLevel(logging.DEBUG)



  @click.command()
  @click.option('--job', '-j', help='Job')
  @click.option('--task', '-t', help='task', type=int, default=0)
  @click.option('--config', '-c', help='config file')
  def cli(job, task, config):
    main(job, task, config)

  if __name__ == "__main__":
    cli()

"""
import numpy as np
# import click
import tensorflow as tf

import pdb
import logging



logging.basicConfig(
    format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)
logger = logging.getLogger('srfapp')



# def task_init(job, task, config=None):
#     t = DistributeReconTask(load_cluster_configs(config))
#     t.cluster_init(job, task)
#     return t


# def create_master_graph(task: DistributeReconTask, x):
#     mg = MasterGraph(x, task.nb_workers(), task.ginfo_master())
#     task.add_master_graph(mg)
#     logger.info("Global graph created.")
#     return mg


# def create_worker_graphs(task: DistributeReconTask, image_info,
#                          data_info: DataInfo):
#     for i in range(task.nb_workers()):
#         logger.info("Creating local graph for worker {}...".format(i))
#         task.add_worker_graph(
#             WorkerGraphLOR(
#                 task.master_graph,
#                 image_info,
#                 {a: data_info.lor_shape(a, i)
#                  for a in ['x', 'y', 'z']},
#                 i,
#                 task.ginfo_worker(i),
#             ))
#     logger.info("All local graph created.")
#     return task.worker_graphs

from ..task import TorTask


class TaskCreator:
    task_list = {{'..task','TorTask'}, }
    def __init__(self):
        pass
    
    def _get_task_name(self, task_config):
        import json
        if isinstance(task_config, str):
            with open(task_config, 'r') as fin:
                c = json.load(fin)
        else:
            print("invalid task config file: {}.".format(task_config))
            raise ValueError
        return c['task_type']

    def _isinlist(self, task_name:str):
        if task_name in TaskCreator.task_list:
            return True
        else:
            print("The task type {} is invalid.".format(task_name))
            raise ValueError

    def _create_instance(self, class_name, *args, **kwargs):
        class_meta = getattr(module_meta, class_name)
        obj = class_meta(*args, **kwargs)
        return obj

    def make_task(self, job, task_index, task_config, distribution_config):
        task_name = self._get_task_name(task_config)
        if isinstance(task_name, str):
            if self._isinlist(task_name):
                return  self._create_instance(task_name, job, task_index, task_config, distribution_config)
        else:
            print("Failed to create a task.")
            raise ValueError
    


def main(job, task_index, task_config, distribution_config=None):
    """
    parse the task config file and create corresponding SRF task.
    """
    if task_config is None:
        task_config = './recon.json'
    logger.info("Start reconstruction job: {}, task_index: {}.".format(
        job, task_index))



    # create the distribute task object
    task = make_task()
    TorTask(job, task_index, task_config, distribution_config)
    
    # start to run the task.
    task.run()

