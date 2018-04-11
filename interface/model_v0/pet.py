from typing import Iterable
from abc import abstractmethod
import json
from tensor import Tensor, Vector3
from medical import PhysicsCartesianVolume,PhysicsCartesian, Detector, Projection,BackProjection, Scatter
from shape import Patch, Box
class EfficiencyMap(PhysicsCartesianVolume):
    """
    由反投影获取的一般图像。
    """
    def __init__(self, data, physicscartesian):
        self.data = data
        self.physicscartesian = physicscartesian
    
    def compile(self):
        
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



class DataProjection(Tensor, BackProjection):
    """
    PET重建中的投影数据
    """    
    def __init__(self, data, detector:'DetectorPET'):
        self.detector:DetectorPET
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
        pass



class PatchPET(DetectorPET):
    """
    由Patch构成的Scanner
    """
    def __init__(self, patches_file, pixelsize:Vector3):
        self.blocks = []
        self.blocks = self._make_blocks(patches_file)
        self.pixelsize = pixelsize

    def _make_blocks(self, patches_file, pixelsize):
        """
        构造探测器块
        """
        for inner_face, outer_face in zip(patches_file):
            patch = Patch(inner_face,outer_face,pixelsize)
            self.blocks.append(patch)
        pass

    def get_lors(self):
        """
        """
        pass

class CylinderPET(DetectorPET):
    """
    由Box构成的Scanner
    """
    def __init__(self, box_origin:Box, num_rings:int, num_blocks_per_ring:int, pixelsize:Vector3):
        self._pixelsize = pixelsize
        self.blocks:Iterable[box] = []
        self._make_blocks(box_origin, num_rings, num_blocks_per_ring)
    
    def _make_blocks(box_origin, num_rings, num_blocks_per_ring):
        pass
    
    def get_lors(self):
        pass

class DetectorPairs(DetectorPET):
    """
    表示探测器对列表，一个detector
    """
    def __init__(self, block_pairs):
        pass
    
    def get_lors(self):
        pass
