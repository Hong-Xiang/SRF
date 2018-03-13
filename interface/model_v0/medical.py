from typing import Iterable
from interface import Interface, PhysicsCartesian
from tensor import Tensor

class Detector(Interface):
    """
    探测器几何基类,决定探测到的数据格式
    """
    def __init__(self, child_dectector:Iterable['Detector']):
        """

        """
        raise NotImplementedError

class PhysicsCartesianVolume(Tensor, PhysicsCartesian):
    """
    笛卡尔坐标系下的离散网格数据
    """
    def __init__(self, data, name = None, grid:PhysicsCartesian = None, shape = 'PhysicsCartesianVolume'):
        pass