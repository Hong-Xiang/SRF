from typing import Tuple, Callable, Dict, Any,Iterable
from abc import ABCMeta
import numpy as np
from tensor import Tensor, Info, Vector2, Vector3

# class Interface(metaclass=ABCMeta):
#     """
#     Interface class, indicating there is no Tensor member in this class.
#     """    
#     pass     


class Discretization(Interface):
    """
    离散化,输入tensor,给定一种离散方式，输出为按目标离散化的tensor
    """
    _fields = ['discretization']
    def __init__(self,discritization:'Cartesian'):
        pass

    def resample_to(self,new_discritization:'Discritization'):
        raise NotImplementedError
    def discritization(self) -> Callable[[Tensor],int] or int:
        pass


# class Dimension(Interface):
#     """
#     维度
#     """
#     # def __init__(self, dim:int):
#     #     self.dim = dim
#     def dim(self) -> Callable[[Any],int] or int:
#         def parse(t:Any):
#             return t.dimension
#         return parse

# Grid = Vector3


# class Grid(Dimension):
#     """
#     网格
#     """
#     def __init__(self, x, y, z):
#         self.x = x
#         self.y = y
#         self.grid = grid
#     def grid(self):
#             self.dim()(owner)
#         return 

# class PixelSize(Interface):
#     """
#     由对象的信息获取pixel size并输出
#     """
#     def pixelsize(self):
#         def parse(oj:Info):
#             info = oj.info
#             return info['pixelsize']
#         return parse
            
# class Position(Dimension):
#     """
#     位置函数，返回对象三维或二维坐标
#     """
#     _fields = ['position']
#     def position(self):
#         return self.dimension_info('position')

# class Orientation(Dimension):
#     """
#     输入tensor，从tensor中的信息获取平面法向量并输出
#     """
#     _fields = ['orientation']
#     def orientation(self):
#         return self.dimension_info('orientation')

# class Size(Dimension):
#     """
#     返回对象的尺寸信息
#     """
    

class Cartesian(Discretization):
    """
    笛卡尔坐标系定义
    """
    def __init__(self,discretization:Discretization, grid:Vector3):
        self.grid = grid
        self.discritization = discretization

class PhysicsCartesian(Cartesian):
    """
    用于图像或探测器网格的定义,包含像素尺寸，位置，以及排布方式
    """
    def __init__(self, discretization:Discretization, grid:Vector3, orientation:Vector3, pixelsize:Vector3, position:Vector3):
        super().__init__(self,discretization,grid)
        self.orientation = orientation
        self.pixelsize = pixelsize
        self.position = position
    
    def get_centers(self) ->Iterable[Vector3]:
        grid = self.grid
        
        grid_size = np.array([grid.x, grid.y,grid.z]
        centers = np.array

    def locate_point(self, point:Vector3):
        
