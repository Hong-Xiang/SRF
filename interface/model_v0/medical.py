from typing import Iterable, Callable
from interface import Interface, PhysicsCartesian
from tensor import Info, Tensor

class PhysicsCartesianVolume(Tensor):
    """
    笛卡尔坐标系下的离散网格数据
    """
    def __init__(self, data, phy_cartesian:PhysicsCartesian):
        self.data = data
        self.PhysicsCartesian
    def _make_info(self):
        pass
    
    def __add__(self, volume) :
        pass
    
    def __sub__(self, volume):
        pass
    def __mul__(self,volume):
        pass

    def __truediv__(self, volume):
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


class Detector(BackProjection):
    """
    探测器几何基类,决定探测到的数据格式
    """
    
    def backproject(self) -> Callable[[Tensor], Tensor]:
        """
        """
        def do(data:Tensor, coodinate: PhysicsCartesian,model):
            # call the tf.op
            return do


    def split(self) ->Callable[[Tensor], Tensor]:
        """

        """
        pass
    


    
    