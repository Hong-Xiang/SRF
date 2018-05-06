#from ..graph.master_sino import MasterGraph
#from ..graph.worker_sino import WorkerGraphSINO

import numpy as np
import tensorflow as tf
import pdb
import logging


logging.basicConfig(
    format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)
logger = logging.getLogger('srfapp')

# from ..task import SinoTask
from ..task import SRFTaskInfo, SinoTaskInfo

class SRFApp():
    @classmethod
    def make_task(cls, job, task_index, task_info:SRFTaskInfo, distribution_config = None):
        return task_info.task_cls(job,task_index, task_info.info, distribution_config)


def main(job, task_index, task_config, distribution_config=None):
    if task_config is None:
        task_config = '/home/twj2417/SRF/SRF/src/python/recon.json'
    logger.info("Start reconstruction job: {}, task_index: {}.".format(
        job, task_index))
        
    import json
    if isinstance(task_config, str):
        with open(task_config, 'r') as fin:
            c = json.load(fin)
    else:
        print("invalid task config file: {}.".format(task_config))
        raise ValueError

    # create the distribute task object
    tc = SinoTaskInfo(c)
    task = SRFApp.make_task(job, task_index, tc, distribution_config)

    task.run()

