from dxl.learn.core import Model


class Projection(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            PROJECTION_DATA = 'projection_data'
            RESULT = 'result'
        
    def __init__(self, info, *, image, projection_data, config):
        """
        Args:
        physics: 
        """
        super().__init__(self, info, inputs={
            self.KEYS.TENSOR.IMAGE: image,
            self.KEYS.TENSOR.PROJECTION_DATA: projection_data
        }, config=config)

    def kernel(self, inputs):
        raise NotImplementedError

class BackProjection(object):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            PROJECTION_DATA = 'projection_data'
            RESULT = 'result'

    def kernel(self, inputs):
        raise NotImplementedError


class Projection(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            # PROJECTION = 'projection'
            # SYSTEM_MATRIX = 'system_matrix'
            # EFFICIENCY_MAP = 'efficiency_map'
            LORS = 'lors'

    def __init__(self,
                 info,
                 *,
                 inputs,
                 physics_model):
        self.grid = np.array(grid, dtype=np.int32)
        self.center = np.array(center, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)
        self.kernel_width = float(kernel_width)
        self.tof_bin = float(tof_bin)
        self.tof_sigma2 = float(tof_sigma2)
        print(tof_bin)
        super().__init__(
            name,
            {
                self.KEYS.TENSOR.IMAGE: image,
                self.KEYS.TENSOR.LORS: lors
            },
            graph_info=graph_info)

    def kernel(self, inputs):
        img = inputs[self.KEYS.TENSOR.IMAGE].data
        grid = self.grid
        center = self.center
        size = self.size
        lors = inputs[self.KEYS.TENSOR.LORS].data
        lors = tf.transpose(lors)
        kernel_width = self.kernel_width
        tof_bin = self.tof_bin
        tof_sigma2 = self.tof_sigma2
        projection_value = projection(
            lors=lors,
            image=img,
            grid=grid,
            center=center,
            size=size,
            kernel_width=kernel_width,
            model=model,
            tof_bin=tof_bin,
            tof_sigma2=tof_sigma2)
        return Tensor(projection_value, None, self.graph_info.update(name=None))


class BackProjection(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            # PROJECTION = 'projection'
            # SYSTEM_MATRIX = 'system_matrix'
            # EFFICIENCY_MAP = 'efficiency_map'
            LORS = 'lors'

    def __init__(self, name, image,
                 grid, center, size,
                 lors,
                 tof_bin, tof_sigma2,
                 kernel_width,
                 graph_info):
        self.grid = np.array(grid, dtype=np.int32)
        self.center = np.array(center, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)
        self.kernel_width = float(kernel_width)
        self.tof_bin = float(tof_bin)
        self.tof_sigma2 = float(tof_sigma2)
        print(tof_bin)
        super().__init__(
            name,
            {
                self.KEYS.TENSOR.IMAGE: image,
                self.KEYS.TENSOR.LORS: lors
            },
            graph_info=graph_info)

    def kernel(self, inputs):
        img = inputs[self.KEYS.TENSOR.IMAGE].data
        grid = self.grid
        center = self.center
        size = self.size
        lors = inputs[self.KEYS.TENSOR.LORS].data
        lors = tf.transpose(lors)
        kernel_width = self.kernel_width
        tof_bin = self.tof_bin
        tof_sigma2 = self.tof_sigma2
        backprojection_image = backprojection(
            lors=lors,
            image=img,
            grid=grid,
            center=center,
            size=size,
            kernel_width=kernel_width,
            model=model,
            tof_bin=tof_bin,
            tof_sigma2=tof_sigma2)
        return Tensor(backprojection_image, None, self.graph_info.update(name=None))
