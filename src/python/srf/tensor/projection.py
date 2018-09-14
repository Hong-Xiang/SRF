from dxl.learn import Tensor


class ProjectionData(Tensor):
    def split_subsets(self, nb_subsets):
        raise NotImplementedError


class LORs(Tensor):
    def __init__(self, model):
        pass

    def backprojection(self, model=None, graph):
        from srf.model import BackProjection
        return BackProjection(model=model, intputs={'input': self}, name=graph.name / 'backprojection')()

    def split_xyz(self, graph):
        from srf.model.preprocessing import ProjectionSplitXYZ
        pass


class LORsInXYZ(LORs):
    def backprojection(self, model):
        pass


class Sinogram3D(Tensor):
    def backprojection(self, model):
        pass
