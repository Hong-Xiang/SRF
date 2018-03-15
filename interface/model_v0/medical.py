from typing import Iterable, Callable
from interface import Interface, PhysicsCartesian
from tensor import Info, Tensor

class PhysicsCartesianVolume(Tensor, PhysicsCartesian):
    """
    笛卡尔坐标系下的离散网格数据
    """
    _fields = ['physicscartesian']
    def __init__(self, data, name = None, grid:PhysicsCartesian = None, shape = 'PhysicsCartesianVolume'):
        pass
    def _make_info(self):
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
    def backproject(self, events:Tensor,coordinate:PhysicsCartesian) -> PhysicsCartesianVolume:

        pass


class Detector(Info, BackProjection):
    """
    探测器几何基类,决定探测到的数据格式
    """
    def __init__(self, child_dectector:Iterable['Detector']):
        """

        """
        raise NotImplementedError
    
    def backproject(self) -> [Callable[Tensor,PhysicsCartesian], Tensor]:
        """
        """
        def do(data:Tensor, coodinate: PhysicsCartesian,model):
            # call the tf.op
            return do


    def split(self) ->[Callable[Tensor], [Tensor]]:
        """

        """
        pass
    


    
    