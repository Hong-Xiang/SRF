from abc import ABCMeta

from .sinodatanew import SinoToRSpec, ReconSpec, MatrixToRSpec
# from task import SRFTaskInfo

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


class SinoTaskInfo(SRFTaskInfo):
    #task_cls = SinoTask
    _fields = {
        'Recon_info',
        'Image_info',
        'Input_info'
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
        self.image = ImageSpec(config['image'])
        self.sino = SinoToRSpec(config['sino'])
        self.matrix = MatrixToRSpec(config['matrix'])
        self.osem = OSEMSpec(config['osem'])

    def to_dict(self):
        result = super().to_dict()
        result.update({
            'image': self.image.to_dict(),
            'sino': self.sino.to_dict(),
            'matrix': self.matrix.to_dict(),
            'osem': self.osem.to_dict(),
        })
        return result