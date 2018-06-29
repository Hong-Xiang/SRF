import numpy as np 

import dxl.learn.core.config import dlcc
from dxl.learn.core import make_distribution_session

from srf.scanner.pet import CylindricalPET
from srf.scanner.pet import MultiPatchPET





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
    def __init__(self, job, task_index, task_config, distribution = None):
        """ initialize the application 


        """
        #set the global configure of this application.
        dlcc.set_global_config(task_config)
        self._scanner = self._make_scanner(task_config)
        self._task = self._make_task(job, self._scanner, task_index, task_config, distribution_config= distribution)
        self.run()

    def _make_scanner(self, task_config):
        """Create a specific scanner object.
        A specific scanner is built according to the user's configuration on scanner.
        The scanner will be then used in the task of this application.

        
        Args:
            task_config: the configuration file of this task.
        
        Returns:
            A scanner object.
        """
        if dlcc.DefaultConfi g

    def _make_task(self, scanner, job, task_index, task_config, distribution_config=None):
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
        



    def run(self):
        """
        Run the task. 
        """
        self._task.run_task()