from typing import Iterable
from abc import abstractmethod
import numpy as np
from interface import Size, Interface
from tensor import Info, Vector3

class Shape(Interface):
    """
    几何抽象类
    """
    @abstractmethod
    def _is_valid(self):
        """
        检查此几何是否有效
        """
        pass


class Polygon(Shape):
    """
    空间中一个三维平面,由一组点构成。
    """
    def __init__(self, points:Iterable[Vector3]):
        self.points = points
    
    def normal(self) -> Vector3:
        num_points = len(self.points)
        p0 = self.points[0]

        return

        
    def is_in_polygon(self, point:Vector3) -> bool:
        """
        判断三维空间中一个点是否在平面内
        """
        pass

    def _is_valid(self) -> bool:
        """
        """
        return 

class Patch(Shape):
    """
    由两个平行面构成的块。
    """
    def __init__(self,inner_face:Polygon, outer_face:Polygon):
        self.inner_face = inner_face
        self.outer_face = outer_face
    def _is_valid(self):
        pass


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

    def valid_centers(self, pixelsize:Vector3) ->Iterable[Vector3]:
        """
        给定晶体大小，获取所有有效晶体位置
        1.计算最小长方体尺寸，参考之前的C++代码
        2.根据晶体大小离散化长方体，并返回在Patch内部的点。
        """
        pass

    def _find_min(self, dectector:'PatchPET') ->Box:
        """
        计算包含此几何体的最小长方体
        """
        pass



class Box(Shape):
    """
    一个带大小尺寸和方位的长方体
    """
    def __init__(self, orientation:Vector3, size:Vector3):
        self.orientation = orientation
        self.size = size
    """
    可以添加一些必要的函数
    """




