from typing import Iterable
from tensor import Tensor, Vector3
from interface import Interface,PhysicsCartesian
from medical import PhysicsCartesianVolume,Detector


class EfficiencyMap(PhysicsCartesianVolume):
    """
    由反投影获取的一般图像。
    """
    def __init__(self):
        pass

class Scatter(Interface):
    """
    to do
    """
    pass

class Projection(Interface):
    """
    PET重建中的投影方法
    """
    def project(self, image:'ImageEmission', events:'DataProjection'):
        pass

class BackProjection(Interface):
    """
    PET重建中的反投影方法
    """
    def backproject(self, events:'DataProjection',image:'ImageEmission'):
        pass


class ImageEmission(PhysicsCartesianVolume,Projection,Scatter):
    """
    用于PET重建的图像
    """
    def __init__(self,data):
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
        pass
    def backprojection(self, image:'ImageEmission', model:'BackProjection'):
        """
        将投影数据反投影到
        """
        pass




class DetectorPET(Detector):
    """
    用于PET探测器
    """
    def get_lors(self) -> 'DataProjection':
        """
        获取此探测器的所有LOR线
        """
        NotImplementedError

class DectectorSplitable(DetectorPET):
    """

    """
    def split(self) -> Iterable[DetectorPET]:
        pass
















