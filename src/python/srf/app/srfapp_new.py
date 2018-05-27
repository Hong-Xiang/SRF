import numpy as np 

from srf.scanner.pet import CylindricalPET
from srf.scanner.pet import MultiPatchPET

from srf.graph.




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
        
        Task: descripts the computing task of this task application. Task  

    """
    def __init__(self, job, task_index, task_config, distribution = None):
        """ create a scanner and the task 
        """
        self._scanner = self._make_scanner(task_config)
        self._task = self._make_task(job, scanner, task_index, task_config, distribution_config= None)

    def _make_scanner(self, task_config):

        pass

    def _make_task(self, scanner, job, task_index, task_config, distribution_config=None):
        pass



    def run(self):


        self._task.run_task()

        pass