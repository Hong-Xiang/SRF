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
from ..task.recon import ReconTask


logging.basicConfig(
    format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)
logger = logging.getLogger('app.reconstruction')



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


def main(job, task_index, task_config, distribution_config=None):
    if task_config is None:
        task_config = './recon.json'
    logger.info("Start reconstruction job: {}, task_index: {}.".format(
        job, task_index))

    # create the distribute task object
    task = ReconTask(job, task_index, task_config, distribution_config)
    
    # start to run the task.
    task.run()

