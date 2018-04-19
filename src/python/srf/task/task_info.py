from abc import ABCMeta

from .tor import TorTask

class SRFTaskInfo(metaclass=ABCMeta):
    task_cls = None
    _fields = {
        'task_type',
        'work_directory'
    }
    def __init__(self, task_configs: dict):
        for a in self._fields:
            if a not in task_configs.keys():
                print("the configure doesn't not have the {} key".format(a))
                raise KeyError
        self.info = task_configs




class TorTaskInfo(SRFTaskInfo):
    task_cls = TorTask
    _fields = {
        'image_info',
        'osem_info',
        'image_info',
        'tof_info'
    }

    def __init__(self, task_configs: dict):
        super().__init__(task_configs)

    
