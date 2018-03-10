from typing import Callable, Iterable
from .low_level import DiscreteVolume, Cartesian, Interface, Multidimensional, Tensor
from abc import abstractmethod


class VectorLowDim(Tensor):
    dim = None
    pass


class Vector2(VectorLowDim):
    dim = 2

    def __init__(self, x, y):
        pass

    @property
    def x(self):
        pass

    @property
    def y(self):
        pass


class Vector3(VectorLowDim):
    dim = 3

    def __init__(self, x, y, z):
        pass

    @property
    def x(self):
        pass

    @property
    def y(self):
        pass

    @property
    def z(self):
        pass


class Position(Interface, Multidimensional):
    def position(self) -> Callable[[Tensor], VectorLowDim]:
        def parse_position(t):
            import json
            info = json.loads(t.info)
            vcls = {
                3: Vector3,
                2: Vector2,
            }[self.dim(t)]
            return vlcs(*info['position'])


class PixelSize(Interface, Multidimensional):
    def pixel_size(self) -> Callable[[Tensor], VectorLowDim]:
        pass


class PhysicsCartesian(Cartesian, Position, PixelSize):
    pass


class PhysicsCartesianDiscreteVolume(DiscreteVolume):
    def __init__(self, data, discretization):
        pass


class CenterPoint(Interface, Multidimensional):
    @abstractmethod
    def center_point(self) -> Callable[[Tensor], VectorLowDim] or VectorLowDim:
        pass


class Detector(Tensor):
    pass


class DetectorIterable(Detector):
    pass


class DetectorCrystalPair(Detector, CenterPoint):
    def __init__(self, data,):
        pass

    def center_point(self):
        return


import tensorflow as tf


class DetectorIterable(Detector):
    def split(self, nb_split) -> Callable[[Tensor], Tensor]:
        def split(t: DataProjection):
            with tf.name_scope('split'):
                splitted_detectors = []  # Split detectors
                splitted_datas: Iterable[tf.Tensor] = []
                return DataProjection(data, detec for data, detec in zip(splitted_datas, splitted_detectors))
        return split


class DataProjection(Tensor):
    def __init__(self, data, detector):
        pass

    def backprojection(self, discretization, model) -> ImageEmission:
        pass

    def split(self, nb_split) -> Iterable[DataProjection]:
        pass


class ModelManager:
    @classmethod
    def set_default(f: Callable[[Any], Any], model):
        pass

    @classmethod
    def get(f: Callable[[Any], Any]):
        pass


class ImageEmission(PhysicsCartesianDiscreteVolume):
    def __init__(self, data, discretization)
        super().__init__(data, discretization)

    def projection(self, detector: Detector, model) -> DataProjection:
        if model is None:
            model = ModelManager.get(__class__.projection)
        return self.projector.projection()(self.data, detector)

    def __truediv__(self, effmap: EfficiencyMap) -> 'ImageEmission':
        if self.discretization != effmap.discretization:
            effmap = effmap.resample_to(self.discretization)
        return ImageEmission(self.data / effmap.data, self.discretization)


class ImageTransmission(PhysicsCartesianDiscreteVolume):
    def __init__(self, data, discretization):
        pass


class EfficiencyMap(PhysicsCartesianDiscreteVolume):
    pass


class Scatter(Interface):
    @abstractmethod
    def scatter(self) -> Callable[[Tensor, Tensor], Tensor]:
        pass


class Projection(Interface):
    @abstractmethod
    def projection(self) -> Callable[[ImageEmission, Detector], Tensor]:
        pass


class ProjectionDetectorPair(Interface):
    @abstractmethod
    def projection(self) -> Callable[[ImageEmission, DetectorCrystalPair], Tensor]:
        def projection(image, detector):
            center = detector.center_point()
