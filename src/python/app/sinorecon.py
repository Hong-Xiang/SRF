from ..graph.master_sino import MasterGraph
from ..graph.worker_sino import WorkerGraphSINO
from ..task.sino import sinoTask

import numpy as np
import tensorflow as tf
import pdb
import logging


logging.basicConfig(
    format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)
logger = logging.getLogger('app.reconstruction')



def main(job, task_index, task_config, distribution_config=None):
    if task_config is None:
        task_config = './recon.json'
    logger.info("Start reconstruction job: {}, task_index: {}.".format(
        job, task_index))

    # create the distribute task object
    task = sinoTask(job, task_index, task_config, distribution_config)
    
    # start to run the task.
    task.run()




