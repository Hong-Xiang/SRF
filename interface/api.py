from typing import Iterable


class Event:
    pass


class Image3D:
    pass


img = Image3D(ndarray)


class WithProjection:
    pass


class Image:
    def __init__(self, project_model):
        # Parse data
        self.data
        self.geoinfo  # Geometry Class

    def projection(self, events: Iterable[Event]):
        project_model =
        pass




class Event:
    def __init__(self, detects: DetectorPair):
        self.dets: DetectorPair
        self.gamma: NumberOfGamma


class LOR:
    def __init__(self, ):
        pass


Projector.projection(Image, DetectorPair)


class Image(ndarray, IProjection, I)


class ImageXXX(Image, IP)


class Projector:
    @classmethod
    def projection(cls, img: Image, events: Iterable[Event])


def projection(image, events):
    """
    From projection.so (TensorFlow custom op)
    """
    pass


def main():
    img = Image()
    events = [Events() for i in range(10)]

    for i in range(100):
        proj = Projector.projection(img, events)
        img = backprojection(proj, events)

    for i in range(100):
        proj = projection_seldon(img, events)
        img = backprojection(proj, events)


def main():
    img = Image.zeros()
    events = [Detector() for i in range(10)]

    proj = img.projection(event)

    Projector.projection(img, events)

    Image3D
    Image2D


class Simulation:
    def compile(self):
        pass


NB_HOSTS = 10

eff_map = Simulation.calculate_efficient_map(Image, Dector)

Simulation.compile()
