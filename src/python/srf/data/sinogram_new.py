from doufo.tensor import Tensor

class SinogramData(Tensor):
    def __init__(self, sinogram):
        super().__init__(sinogram)

    @property
    def nb_view(self):
        return self.shape[0]

