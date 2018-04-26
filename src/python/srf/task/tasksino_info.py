from abc import ABCMeta

#from .sino import SinoTask
from task import SRFTaskInfo,SinoTask

# class SRFTaskInfo(metaclass=ABCMeta):
#     task_cls = None
#     _fields = {
#         'task_type',
#         'work_directory'
#     }
#     def __init__(self, task_configs: dict):
#         for a in self._fields:
#             if a not in task_configs.keys():
#                 print("the configure doesn't not have the {} key".format(a))
#                 raise KeyError
#         self.info = task_configs


class SinoTaskInfo(SRFTaskInfo):
    task_cls = SinoTask
    _fields = {
        'Recon_info',
        'Image_info',
        'Input_info'
    }

    def __init__(self, task_configs: dict):
        super().__init__(task_configs)

    
