from dxl.learn.core import Graph, NoOp

from .master import MasterGraph
from .worker import WorkerGraph
import tensorflow as tf


class LocalReconstructionGraph(Graph):
    def __init__(self, info, master_data_loader, worker_data_loader, *, config=None, nb_iteration=10):
        self._master_data_loader = master_data_loader
        self._worker_data_loader = worker_data_loader
        config = self._parse_input_config(config, {
            'nb_iteration': nb_iteration,
        })
        super().__init__(info, config=config)

    def kernel(self):
        m = self.subgraphs['master'] = MasterGraph(self.info.child_scope('master'),
                                                   loader=self._master_data_loader, nb_workers=1)
        self.tensors['x'] = m.tensor('x')
        w = self.subgraphs['worker'] = WorkerGraph(self.info.child_scope('worker'), m.tensor(
            'x'), m.tensor(m.KEYS.TENSOR.BUFFER)[0], loader=self._worker_data_loader)
        with tf.control_dependencies([m.tensor['init'].data, w.tensor['init'].data]):
            self.tensors['init'] = NoOp()
        self.tensors['recon'] = w.tensor('update')
        self.tensors['update'] = m.tensor('update')

    def run(self, sess):
        sess.run(self.tensor('init'))
        for i in range(self.config('nb_iterations')):
            sess.run(self.tensor('recon'))
            sess.run(self.tensor('update'))
            x = sess.run(self.tensor('x'))
            np.save('recon_{}.npy'.format(i), x)
