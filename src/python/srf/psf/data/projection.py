import h5py
import numpy as np
from scipy import sparse

from srf2.meta.projection_meta import *

__all__ = ('Projection', 'Sinogram_projection', 'Listmode_projection')


class Projection:
    def __init__(self, data = None, meta: Projection_meta = None):
        self._meta = meta
        self._data = data

    @property
    def data(self):
        return self._data

    @property
    def meta(self):
        return self._meta

    def fmap(self, f):
        return Projection(f(self._data), self.meta)

    def transpose(self):
        pass

    def transfer(self, data_type = None):
        raise NotImplementedError

    def save_h5(self, path):
        raise NotImplementedError

    def load_h5(path):
        raise NotImplementedError


class Listmode_projection(Projection):
    def __init__(self, data: np.ndarray, meta: Listmode_projection_meta):
        if data is None:
            data = np.array([])
        if meta is None:
            meta = Listmode_projection_meta()
        super().__init__(data, meta)

    @property
    def fst(self):
        return self.data[:, :3]

    @property
    def snd(self):
        return self.data[:, 3:6]

    @property
    def vals(self):
        return self.data[:, 6]

    def save_h5(self, path = None, mode = None):
        if path is None:
            path = 'tmp' + self.__class__.__name__ + '.h5'
        self.meta.save_h5(path, 'mode')
        with h5py.File(path, 'r+') as fout:
            group = fout.create_group('Listmode_projection_data')
            group.create_dataset('_data', data = self.data, compression = "gzip")

    @classmethod
    def load_h5(cls, path = None):
        if path is None:
            path = 'tmp' + cls.__name__ + '.h5'
        meta = Listmode_projection_meta.load_h5(path)
        with h5py.File(path, 'r') as fin:
            group = fin['Listmode_projection_data']
            data = np.array(group['_data'])
            return Listmode_projection(data, meta)

    def transfer(self, data_type = None):
        if data_type == 'sinogram':
            data2 = _listmode_to_sinogram(self.data, self.meta)
            return Sinogram_projection(data2, self.meta.transfer('sinogram'))
        elif data_type == 'listmode':
            return self
        else:
            raise NotImplementedError


class Sinogram_projection(Projection):
    def __init__(self, data: sparse.csr_matrix, meta: Sinogram_projection_meta):
        if meta is None:
            meta = Sinogram_projection_meta()
        if data is None:
            data = sparse.csr_matrix((meta.n_sensors_all, meta.n_sensors_all), dtype = np.float32)
        super().__init__(data, meta)

    @property
    def fst(self):
        return self.data.nonzero()[0]

    @property
    def snd(self):
        return self.data.nonzero()[1]

    @property
    def vals(self):
        return self.data.data

    def save_h5(self, path = None, mode = 'w'):
        if path is None:
            path = 'tmp' + self.__class__.__name__ + '.h5'
        self.meta.save_h5(path, 'w')
        with h5py.File(path, 'r+') as fout:
            group = fout.create_group('Sinogram_projection_meta')
            group.create_dataset('fst', data = self.fst, compression = "gzip")
            group.create_dataset('snd', data = self.snd, compression = "gzip")
            group.create_dataset('vals', data = self.vals, compression = "gzip")

    def load_h5(cls, path = None):
        if path is None:
            path = 'tmp' + cls.__name__ + '.h5'
        meta = Sinogram_projection_meta.load_h5(path)
        with h5py.File(path, 'r') as fin:
            group = fin['projection_data']
            fst = np.array(group['fst'])
            snd = np.array(group['snd'])
            vals = np.array(group['vals'])
            csr = sparse.csr_matrix((vals, (fst, snd)), shape = (meta.n_sensors_all,
                                                                 meta.n_sensors_all))
            return Sinogram_projection(csr, meta)

    def transfer(self, data_type = None):
        if data_type == 'listmode':
            data2 = _sinogram_to_listmode(self.data, self.meta)
            return Listmode_projection(data2, self.meta.transfer('listmode'))
        elif data_type == 'sinogram':
            return self
        else:
            raise NotImplementedError


def _listmode_to_sinogram(listmode: np.ndarray, meta: Projection_meta):
    angle_per_block = np.pi * 2 / meta.nb_blocks_per_ring
    x_data1 = listmode[:, 0]
    y_data1 = listmode[:, 1]
    z_data1 = listmode[:, 2]
    angles1 = np.arctan2(y_data1, x_data1) + np.pi
    iblock1 = (np.round(angles1 / angle_per_block) % meta.nb_blocks_per_ring).astype(int)
    angle = iblock1 * angle_per_block
    y_data1o = -x_data1 * np.sin(angle) + y_data1 * np.cos(angle)
    y_ind1 = np.round(
        (y_data1o + meta.size[1] / 2 - meta.unit_size[1] / 2) / meta.unit_size[1]).astype(int)
    z_ind1 = np.round(
        (z_data1 + meta.size[2] * meta.nb_rings / 2 - meta.unit_size[2] / 2) / meta.unit_size[
            2]).astype(int)
    iring1 = np.floor(z_ind1 / meta.shape[2]).astype(int)
    z_ind1 //= meta.shape[2]

    x_data2 = listmode[:, 3]
    y_data2 = listmode[:, 4]
    z_data2 = listmode[:, 5]
    angles2 = np.arctan2(y_data2, x_data2) + np.pi
    iblock2 = (np.round(angles2 / angle_per_block) % meta.nb_blocks_per_ring).astype(int)
    angle = iblock2 * angle_per_block
    y_data2o = -x_data2 * np.sin(angle) + y_data2 * np.cos(angle)
    y_ind2 = np.round(
        (y_data2o + meta.size[1] / 2 - meta.unit_size[1] / 2) / meta.unit_size[1]).astype(
        int)
    z_ind2 = np.round(
        (z_data2 + meta.size[2] * meta.nb_rings / 2 - meta.unit_size[2] / 2) / meta.unit_size[
            2]).astype(
        int)
    iring2 = np.floor(z_ind2 / meta.shape[2]).astype(int)
    z_ind2 //= meta.shape[2]

    n1 = meta.shape[1]
    n2 = meta.shape[2] * n1
    n3 = meta.nb_blocks_per_ring * n2
    n4 = meta.nb_rings * n3
    ind1 = (y_ind1 + n1 * z_ind1 + n2 * iblock1 + n3 * iring1).astype(int)
    ind2 = (y_ind2 + n1 * z_ind2 + n2 * iblock2 + n3 * iring2).astype(int)
    lil_mat = sparse.lil_matrix((n4, n4), dtype = np.float32)
    for i in range(ind1.size):
        lil_mat[ind1[i], ind2[i]] += listmode[i, 6]
        lil_mat[ind2[i], ind1[i]] += listmode[i, 6]
    return lil_mat.tocsr()


def _sinogram_to_listmode(sino: sparse.csr_matrix, meta: Projection_meta):
    rows0, cols0 = sino.nonzero()
    rows, cols = rows0[rows0 > cols0], cols0[rows0 > cols0]
    vals = sino.data[rows0 > cols0]

    x1_0 = x2_0 = (meta.inner_radius + meta.outer_radius) / 2
    y_ind1 = (rows % meta.shape[1]).astype(int)
    y1_0 = (y_ind1 + 0.5) * meta.unit_size[1] - meta.size[1] / 2
    z_ind1 = (rows // meta.shape[1] % meta.shape[2]).astype(int)
    iblock1 = (rows // meta.shape[1] // meta.shape[2] % meta.nb_blocks_per_ring).astype(int)
    iring1 = (rows // meta.shape[1] // meta.shape[2] // meta.nb_blocks_per_ring).astype(int)

    y_ind2 = (cols % meta.shape[1]).astype(int)
    y2_0 = (y_ind2 + 0.5) * meta.unit_size[1] - meta.size[1] / 2
    z_ind2 = (cols // meta.shape[1] % meta.shape[2]).astype(int)
    iblock2 = (cols // meta.shape[1] // meta.shape[2] % meta.nb_blocks_per_ring).astype(int)
    iring2 = (cols // meta.shape[1] // meta.shape[2] // meta.nb_blocks_per_ring).astype(int)
    angle_per_block = np.pi * 2 / meta.nb_blocks_per_ring

    data2 = np.zeros((vals.size, 7))

    data2[:, 0] = x1_0 * np.cos(iblock1 * angle_per_block) - y1_0 * np.sin(
        iblock1 * angle_per_block)
    data2[:, 1] = x1_0 * np.sin(iblock1 * angle_per_block) + y1_0 * np.cos(
        iblock1 * angle_per_block)
    data2[:, 2] = (0.5 + z_ind1) * meta.unit_size[2] + iring1 * meta.size[2] - meta.nb_rings * \
                  meta.size[2] / 2

    data2[:, 3] = x2_0 * np.cos(iblock2 * angle_per_block) - y2_0 * np.sin(
        iblock2 * angle_per_block)
    data2[:, 4] = x2_0 * np.sin(iblock2 * angle_per_block) + y2_0 * np.cos(
        iblock2 * angle_per_block)
    data2[:, 5] = (0.5 + z_ind2) * meta.unit_size[2] + iring2 * meta.size[2] - meta.nb_rings * \
                  meta.size[2] / 2

    data2[:, 6] = vals
    return data2
