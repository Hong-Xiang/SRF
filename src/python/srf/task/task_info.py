from abc import ABCMeta

# from .tor import TorTask
from .data import ToFSpec, ImageSpec, OSEMSpec


class SRFTaskInfo(metaclass=ABCMeta):
    task_cls = None
    _fields = {
        'task_type',
        'work_directory'
    }

    def __init__(self, task_configs: dict):
        for a in self._fields:
            if a not in task_configs.keys():
                raise KeyError(
                    "the configure doesn't not have the {} key".format(a))

        self.info = task_configs


class TorTaskInfo(SRFTaskInfo):
    # task_cls = TorTask
    _fields = {
        'image_info',
        'osem_info',
        'tof_info'
    }

    def __init__(self, task_configs: dict):
        super().__init__(task_configs)


class SRFTaskSpec:
    # task_cls = None
    task_type = None

    def __init__(self, config):
        self.work_directory = config['work_directory']

    def to_dict(self):
        return {'work_directory': self.work_directory}


from .data import ToRSpec, LoRsToRSpec


class ToRTaskSpec(SRFTaskSpec):
    # task_cls = TorTask
    task_type = 'TorTask'

    def __init__(self, config):
        super().__init__(config)
        self.image = ImageSpec(config['image'])
        self.lors = LoRsToRSpec(config['lors'])
        self.tof = ToFSpec(config['tof'])
        self.osem = OSEMSpec(config['osem'])
        self.tor = ToRSpec(config['tor'])

    def to_dict(self):
        result = super().to_dict()
        result.update({
            'image': self.image.to_dict(),
            'lors': self.lors.to_dict(),
            'tof': self.tof.to_dict(),
            'osem': self.osem.to_dict(),
            'tor': self.tor.to_dict()
        })
        return result
