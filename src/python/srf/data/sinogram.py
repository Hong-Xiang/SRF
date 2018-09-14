from doufo.tensor import Tensor


class PETSinogram(Tensor):
    def __init__(self, sinogram):
        super().__init__(sinogram)

    @property
    def nb_view(self):
        return self.shape[0]


class PETSinogram3D(Tensor):
    def __init__(self, sinograms):
        super().__init__(sinograms)

    @property
    def nb_sinogram2d(self) -> int:
        return self.shape[0]

    @property
    def nb_view(self):
        return self.shape[1]

