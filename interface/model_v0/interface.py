from typing import Tuple, Callable, Dict, Any,Iterable
from abc import ABCMeta,abstractmethod
import numpy as np
from tensor import Vector2,Vector3,Tensor
from shape import Box

class Interface(metaclass=ABCMeta):
    """
    Interface class, indicating there is no Tensor member in this class.
    """    
    pass


class Discretization(Interface):
    """
    离散化,输入tensor,给定一种离散方式，输出为按目标离散化的tensor
    """
    def __init__(self,discritization:'Cartesian'):
        pass

    def resample_to(self,new_discritization:'Discritization'):
        raise NotImplementedError
    def discritization(self) -> Callable[[Tensor],int] or int:
        pass


class Dimension(Interface):
    """
    返回对象维度
    """
    #def dim(self) -> int:
    def dim(self) -> Callable[[Tensor],int] or int:
        raise NotImplementedError




class DimensionInfo(Dimension):
    """
    返回对象的含维度信息的属性。
    """
    def dimension_info(self,attr:str):
        def parse(self, t):
            import json
            info = json.loads(t.info)
            vcls = {
                3:Vector3,
                2:Vector2,
            }[self.dim()(t)]
            return vcls(*info[attr])
        return parse


class Grid(DimensionInfo):
    """
    
    """
    def grid(self):
        return self.dimension_info('grid')


class PixelSize(DimensionInfo):
    """
    输入tensor，从tensor中的信息获取pixel size并输出
    """
    def pixelsize(self):
        return self.dimension_info('pixelsize')
            
class Position(DimensionInfo):
    """
    位置函数，返回对象三维或二维坐标
    """
    def position(self):
        return self.dimension_info('position')

class Orientation(DimensionInfo):
    """
    输入tensor，从tensor中的信息获取平面法向量并输出
    """
    def orientation(self):
        return self.dimension_info('orientation')

class Size(DimensionInfo):
    """
    返回对象的尺寸信息
    """
    def size(self):
        return self.dimension_info('size')

class Cartesian(Discretization,Grid,PixelSize,Box):
    """
    笛卡尔坐标系定义，长宽高的定义方式
    """
    def cartesian(self):
        def parse_cartesian(t):
            raise NotImplementedError
        return parse_cartesian

class PhysicsCartesian(Cartesian,PixelSize,Position):
    """
    用于图像或探测器的定义,包含像素尺寸，位置，以及排布方式
    """
    # def __init__(self):
    #     NotImplementedError

    def physics_cartesian(self):
        """
        返回一个tensor的坐标系信息。
        """
        raise NotImplementedError