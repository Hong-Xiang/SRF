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

from typing import Iterable
import pdb
import time

import json
from tqdm import tqdm


from dxl.learn.core import Master, Barrier, ThisHost, ThisSession, Tensor
from dxl.learn.core import load_cluster_configs


from ..graph.master import MasterGraph
from ..graph.worker import WorkerGraphLOR
from ..services.utils import ImageInfo, DataInfo, load_data, logger, print_tensor
from ..services.utils import debug_tensor, load_reconstruction_configs, load_local_data
from ..preprocess.preprocess import partition as preprocess_tor

from ..distribute.recon import DistributeReconTask

# def task_init(job, task, config=None):
#     t = DistributeReconTask(load_cluster_configs(config))
#     t.cluster_init(job, task)
#     return t


def create_master_graph(task: DistributeReconTask, x):
    mg = MasterGraph(x, task.nb_workers(), task.ginfo_master())
    task.add_master_graph(mg)
    logger.info("Global graph created.")
    return mg


def create_worker_graphs(task: DistributeReconTask, image_info,
                         data_info: DataInfo):
    for i in range(task.nb_workers()):
        logger.info("Creating local graph for worker {}...".format(i))
        task.add_worker_graph(
            WorkerGraphLOR(
                task.master_graph,
                image_info,
                {a: data_info.lor_shape(a, i)
                 for a in ['x', 'y', 'z']},
                i,
                task.ginfo_worker(i),
            ))
    logger.info("All local graph created.")
    return task.worker_graphs


def bind_local_data(data_info, task: DistributeReconTask, task_index=None):
    if task_index is None:
        task_index = ThisHost.host().task_index
    if ThisHost.is_master():
        logger.info("On Master node, skip bind local data.")
        return
    else:
        logger.info(
            "On Worker node, local data for worker {}.".format(task_index))
        emap, lors = load_local_data(data_info, task_index)
        task.worker_graphs[task_index].assign_efficiency_map_and_lors(
            emap, lors)


def run_and_save_if_is_master(x, path):
    if ThisHost.is_master():
        if isinstance(x, Tensor):
            x = x.data
        result = ThisSession.run(x)
        np.save(path, result)


def main(job, task_index, config=None, distribution_config=None):
    if config is None:
        config = './recon.json'
    logger.info("Start reconstruction job: {}, task_index: {}.".format(
        job, task_index))

    # create the distribute task object
    task = DistributeReconTask(distribution_config)
    task.cluster_init(job, task_index)
    # task = task_init(job, task_index, distribution_config)

    # load the task configuration
    image_info, data_info, lors_info= load_reconstruction_configs(config)
    logger.info("Local data_info:\n" + str(data_info))

    # make master and worker graphs
    create_master_graph(task, np.ones(image_info.grid, dtype=np.float32))
    create_worker_graphs(task, image_info, data_info)

    # bind the data used on workers
    bind_local_data(data_info, task)

    # add the calculation dependency
    task.make_recon_task()

    task.run_step_of_this_host(task.steps[task.KEYS.STEPS.INIT])
    logger.info('STEP: {} done.'.format(task.steps[task.KEYS.STEPS.INIT]))
    nb_steps = 10
    for i in tqdm(range(nb_steps), ascii=True):
        
        task.run_step_of_this_host(task.steps[task.KEYS.STEPS.RECON])
        logger.info('STEP: {} done.'.format(task.steps[task.KEYS.STEPS.RECON]))

        task.run_step_of_this_host(task.steps[task.KEYS.STEPS.MERGE])
        logger.info('STEP: {} done.'.format(task.steps[task.KEYS.STEPS.MERGE]))

        run_and_save_if_is_master(
            task.master_graph.tensor('x'),
            './debug/mem_lim_result_{}.npy'.format(i))
    logger.info('Recon {} steps done.'.format(nb_steps))
    time.sleep(5)
