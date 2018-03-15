from typing import Iterable
import json
from tensor import Tensor
from medical import PhysicsCartesianVolume,PhysicsCartesian, Detector, Projection,BackProjection, Scatter

class EfficiencyMap(PhysicsCartesianVolume):
    """
    由反投影获取的一般图像。
    """
    def __init__(self):
        pass


class ImageEmission(PhysicsCartesianVolume,Projection,Scatter):
    """
    用于PET重建的图像
    """
    def __init__(self,data):
        pass
    
    def _make_info(self):
        pass

    def projection(self, lors:'DataProjection', model:'Projection'):
        """
        将图像投影到探测器上，得到投影数据
        """
        pass

    def __truediv__(self, effmap:EfficiencyMap):
        pass

class DataProjection(Tensor, BackProjection):
    """
    PET重建中的投影数据
    """    
    def __init__(self, data):
        self.detector:Detector
        self.data = data
    def backprojection(self, coordinate:PhysicsCartesian, model:'BackProjection'):
        """
        将投影数据反投影到
        """
        return self.detector.backproject()(self.data, coordinate, model)




class DetectorPET(Detector):
    """
    用于PET探测器
    """
    def get_lors(self) -> 'DataProjection':
        """
        获取此探测器的所有LOR线
        """
        pass

class PatchPET(DetectorPET):
    """
    由Patch构成的Scanner
    """
    def __init__(self, patches):
        self.patchs = json.load(patches)

    def get_lors(self):
















