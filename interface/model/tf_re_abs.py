from typing import Dict

from .low_level import Tensor, CompileAble
from .mid_level import ImageEmission, DataProjection, DetectorCrystalPair,

class Host:
    def __init__(self, cluster: str, index: int):
        pass


def configurable():
    pass


class Graph(CompileAble):
    @configurable
    def __init__(inputs=Dict[str, Tensor], host: Host, name: str):
        for k in inputs:
            self.inputs[k] = inputs[k]
            self.ts['inputs/' + k] = inputs[k]
        self.host = host
        self.compile(inputs)

    def add_inputs(self, tensors):
        pass

    def add_outputs(self, tensors):
        pass

    def __call__(self, inputs=None):
        self.compile(inputs)

    def kernel(self, inputs=None):
        pass

    def compile(self, inputs=None):
        # 在这里处理tf的scope问题
        self.kernel(inputs)




class ReconstructionStep(Graph):
    class TKeys:
        IMAGE = 'image'
        PROJECTION = 'projection'
        BACK_PROJECTION = 'back_projection'

    def __init__(self, image: ImageEmission, projection: DataProjection, discretization=None, detector: 'Detector description'=None, name='reconstruction_iteration_step'):
        super().__init__({
            self.TKeys.IMAGE: image,
            self.TKeys.PROJECTION: projection
        })
        if discretization is None:
            discretization = self.image.discretization
        self.discretization = discretization
        if detector is None:
            detector = self.events.detector
        self.detector = detector

    def kernel(self, inputs):
        img: ImageEmission = self.ts[self.TKeys.IMAGE]
        proj = img.projection(self.detector)
        proj = proj / self.inputs[self.TKeys.PROJECTION]
        bpi = proj.backprojection(self.discretization)
        return {self.TKeys.BACK_PROJECTION: bpi}


def foo(x, y):
    return x + y

image = ImageEmission()

Host('master', 0)

for i in range(10):
    recon_g = ReconstructionStep(image, sinogram, host=Host('worker', i))(image, sinogram)

for i in range(10):
    recon_g = reconstruction_step(image, sinogram)

image = image + 1

a, b = b, a


