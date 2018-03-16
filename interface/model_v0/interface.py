from typing import Tuple, Callable, Dict, Any,Iterable
from abc import ABCMeta
import numpy as np
from tensor import Tensor, Info, Vector2, Vector3
from enum import Enum
import math
class Interface(metaclass=ABCMeta):
    """
    Interface class, indicating there is no Tensor member in this class.
    """    
    pass     

class Axis(Enum):
    x = 'x'
    y = 'y'
    z = 'z'

class Rotation:
    """
    旋转，可用于旋转三维空间中的一个点
    """
    def __init__(self):
        self._rotation_matrix  = np.matrix(np.zeros([3,3]))

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

    def set_rotation_matrix(self, radian, axis:Axis):
        if axis is Axis.x:
            self._rotation_matrix[0,0] = 1
            self._rotation_matrix[1,1] = math.cos(radian)
            self._rotation_matrix[2,2] = math.cos(radian)
            self._rotation_matrix[1,2] = math.sin(radian)
            self._rotation_matrix[2,1] = -math.sin(radian)
        elif axis is Axis.y:
            self._rotation_matrix[1,1] = 1
            self._rotation_matrix[0,0] = math.cos(radian)
            self._rotation_matrix[2,2] = math.cos(radian)
            self._rotation_matrix[0,2] = -math.sin(radian)
            self._rotation_matrix[2,0] = math.sin(radian)
        elif axis is Axis.z:
            self._rotation_matrix[2,2] = 1
            self._rotation_matrix[0,0] = math.cos(radian)
            self._rotation_matrix[1,1] = math.cos(radian)
            self._rotation_matrix[0,1] = math.sin(radian)
            self._rotation_matrix[1,0] = -math.sin(radian)
        else:
            raise ValueError
    def reset(self):
        self._rotation_matrix = np.matrix(np.zeros([3,3]))
    
    def rotate(self, pt:Vector3):
        rm = self._rotation_matrix
        pt_a = pt.value
        pt_a.reshape((-1,1))
        rpt_m = pt_a * rm
        rpt_a = rpt_m.reshape((1,-1))
        return Vector3(rpt_a[0,0],rpt_a[0,1], rpt_a[0,2])



class Discretization(Interface):
    """
    离散化,输入tensor,给定一种离散方式，输出为按目标离散化的tensor
    """
    def __init__(self,discritization:'Cartesian' = None):
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
    

class Cartesian():
    """
    笛卡尔坐标系定义
    """
    def __init__(self, discretization:Discretization, grid:Vector3):
        self._grid = grid
        self._discritization = discretization
    
    @property
    def grid(self):
        return self._grid
    
    @property
    def discritization(self):
        return self._discritization

class PhysicsCartesian(Cartesian):
    """
    用于图像或探测器网格的定义,包含像素尺寸，位置，以及排布方式
    """
    def __init__(self, discretization:Discretization, grid:Vector3, orientation:Vector3, pixelsize:Vector3, position:Vector3):
        super().__init__(discretization, grid)
        self._orientation = orientation
        self._pixelsize = pixelsize
        self._position = position

        # self._bottom_bound:
        # self._offset:
        #rotate the box

    @property
    def orientation(self):
        return self._orientation
    
    @property
    def pixelsize(self):
        return self._pixelsize
    
    @property
    def position(self):
        return self._position


    def get_centers(self) ->Iterable[Vector3]:
        grid = self.grid
        
        

    def locate_point(self, point:Vector3):
        pass
    def get_bottom_bound(self):
        pass

        
