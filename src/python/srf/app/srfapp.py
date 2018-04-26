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
import json

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)
logger = logging.getLogger('srfapp')


# TODO: Implement one unified config system for all applications.


def _load_config(path):
    """
    Helper function to load .json config file.

    Arguments:
    - `path` path to config file.

    Return:

    - dict of config

    Raises:
    - None

    """
    # TODO: rework to use dxl.fs
    from pathlib import Path
    import json
    with open(Path(path)) as fin:
        return json.load(fin)


def _load_config_if_not_dict(config):
    """
    Unified config_file(via path) and config dict.
    """
    if not isinstance(config, dict):
        return _load_config(config)
    return config


from ..task import TorTask
from ..task import SRFTaskInfo, TorTaskInfo
from ..task.task_info import ToRTaskSpec


class SRFApp():
    """
    Scalable reconstruction framework high-level API. With following methods:
    """

    @classmethod
    def make_task(cls,
                  job,
                  task_index,
                  task_info: SRFTaskInfo,
                  distribution_config=None):
        return task_info.task_cls(job, task_index, task_info.info,
                                  distribution_config)

    @classmethod
    def reconstruction(cls, job, task_index, task_config, distribute_config):
        """
        Distribute reconstruction main entry. Call this function in different processes.
        """
        # task_config = _load_config_if_not_dict(task_config)
        # distribute_config = _load_config_if_not_dict(distribute_config)
        logging.info("Task config: {}.".format(task_config))
        logging.info("Distribute config: {}.".format(distribute_config))
        # task_info = TorTaskInfo(task_config)
        task_spec = ToRTaskSpec(task_config)
        task = task_spec.task_cls(
            job, task_index, task_spec, distribute_config)
        task.run()

    @classmethod
    def efficiency_map_single_ring(cls, job, task_index, task_config, distribute_config):
        pass

    @classmethod
    def efficiency_map_merge(cls, task_config):
        pass

    @classmethod
    def make_tor_lors(cls, config):
        """
        Preprocessing data for TOR model based reconstruction.
        """
        from ..task.task_info import ToRTaskSpec
        from ..preprocess._tor import process
        ts = ToRTaskSpec(config)
        process(ts)

    @classmethod
    def tor_osem_auto_config(cls, recon_config, distribute_config, output=None):
        from dxl.learn.core.distribute import load_cluster_configs
        distribute_config = load_cluster_configs(distribute_config)
        nb_workers = distribute_config.get('nb_workers',
                                           len(distribute_config['worker']))
        from ..task.task_info import ToRTaskSpec
        ts = ToRTaskSpec(recon_config)
        nb_subsets = ts.osem.nb_subsets
        import h5py
        with h5py.File(ts.lors.path_file, 'r') as fin:
            lors = fin[ts.lors.path_dataset]
