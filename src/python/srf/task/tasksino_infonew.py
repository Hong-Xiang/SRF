from abc import ABCMeta

# from .tor import TorTask
from .sinodatanew import  ImageSpec, ReconSpec, SinoToRSpec, MatrixToRSpec


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


class SinoTaskInfo(SRFTaskInfo):
    # task_cls = TorTask
    _fields = {
        'image_info',
        'osem_info',
        'input_info'
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


class SinoTaskSpec(SRFTaskSpec):
    # task_cls = TorTask
    task_type = 'SinoTask'

    def __init__(self, config):
        super().__init__(config)
        self.image = ImageSpec(config['output']['image'])
        self.sino = SinoToRSpec(config['input']['sino'])
        self.matrix = MatrixToRSpec(config['input']['sm'])
        self.recon = ReconSpec(config['algorithm']['recon']['mlem'])


    def to_dict(self):
        result = super().to_dict()
        result.update({
            'image': self.image.to_dict(),
            'sino': self.sino.to_dict(),
            'matrix': self.matrix.to_dict(),
            'recon': self.recon.to_dict()
        })
        return result
