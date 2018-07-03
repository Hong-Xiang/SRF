from srf.graph.reconstruction.local_reconstruction import LocalReconstructionGraph
from srf.data.loader import MasterLoader, WorkerLoader
from dxl.learn.core import SubgraphMakerTable
from srf.model.projection import ProjectionToR
from srf.model.backprojection import BackProjectionToR

from dxl.core.debug import enter_debug
from dxl.learn.core.config import update_config
from dxl.learn.core import Session

enter_debug()


# lambda p, k, x, y: Graph(p, k, x)


# def some_constructor(p, k, x, y):
#     pass


# def some_constructor2(p, k, x, y):
#     pass


# class SomeConstructor:
#     def __init__(self, p, k, x, y):
#         pass


# class C2(SomeConstructor):
#     pass


# class C3(SomeConstructor):
#     pass


# MainGraph(subgraphs=)
# MainGraph2


# def kernel(self):
#     x = ...
#     y = self.subgraph('sub', lambda p, k, x: )

RESOURCE_ROOT = '/mnt/gluster/CustomerTests/SRF/recostruction/run0605/'


def main():
    SubgraphMakerTable.register(
        'reconstruction/worker/reconstruction/projection', ProjectionToR())
    SubgraphMakerTable.register(
        'reconstruction/worker/reconstruction/backprojection', BackProjectionToR)
    update_config('reconstruction/worker/reconstruction',
                  {'center': [0.0, 0.0, 0.0]})
    update_config('reconstruction/worker/reconstruction',
                  {'size': [666.9, 666.9, 1422.72]})
    update_config('projection_model', {
        'tof_sigma2': 162.30,
        'tof_bin': 6.0,
        'kernel_width': 3.86,
    })
    ml = MasterLoader(shape=[195, 195, 416])
    ll = WorkerLoader(RESOURCE_ROOT + 'lors_debug.npz',
                      RESOURCE_ROOT + 'summap.npy')
    g = LocalReconstructionGraph('reconstruction', ml, ll, nb_iteration=10)
    sess = Session()
    g.run(sess)


def a_propsed_main(config):
    g = LocalReconstructionGraph('reconstruction',
                                 MasterGraph('master', loader=MasterLoader(
                                     'loader', shape=[195, 195, 416])),
                                 WorkerGraph('worker', loader=WorkerLoader('loader'),
                                             recon_step=ReconStep('recon', graphs={
                                                 'projection': ProjectionSiddonToR(),
                                                 'backprojection': BackProjectionToR()
                                             })))
    g.make()
    sess = Session()
    g.run(sess)

    s = Stacked([Conv2D(filters=32) for i in range(10)])

    s = Stacked([Residual(xxx) for i in range(10)])
    s = Stacked([Incept(xxx) for i in range(10)])


if __name__ == "__main__":
    main()
