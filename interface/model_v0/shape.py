from abc import abstractmethod, ABCMeta
from typing import Iterable

from tensor import Vector3
from interface import Interface, Size


class Shape(Interface):
    """
    几何形状
    """
    
    @abstractmethod
    def is_valid(self):
        """
        检查此几何是否有效
        """
        NotImplementedError


class Polygon(Shape):
    """
    空间中一个三维平面,由一组点构成。
    """    
    def normal(self):
        """
        返回法向量
        """ 
        pass
    
    def is_in_polygon(self, point:Vector3):
        """
        判断三维空间中一个点是否在平面内
        """
        pass

class Box(Shape,Size):
    """
    一个带大小尺寸的长方体
    """


class Patch(Shape):
    """
    由两个平行面构成的块。
    """
    def __init__(self,inner_face:Polygon, outer_face:Polygon, name = None,shape = 'Patch'):
        NotImplementedError
    
    def normal(self) ->Vector3:
        """
        返回法向量
        """
        pass
    
    def is_in_patch(self, point:Vector3):
        """
        判断三维一个点是否在此Patch中
        """   
        pass
    
    def min_box(self):
        """
        计算包含此几何体的最小长方体
        """
        pass