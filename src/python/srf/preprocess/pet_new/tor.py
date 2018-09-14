from ..task.data import TorInfo
import logging
import numpy as np
from typing import Dict
# from ..preprocess.preprocess import preprocess, cut_lors
from dxl.data import NDArrayHDF5 as ArrayNDHDF5
from dxl.data.io import save_h5
from pathlib import Path

from ..task.data import ToFSpec, LoRsToRSpec

XYZ = ['x', 'y', 'z']


def p0(arr):
    return arr[:, 0:3]


def p1(arr):
    return arr[:, 3:6]


def t_dis(arr):
    return arr[:, 6]


def extra(arr):
    return arr[:, 7:]


def axis2int(axis):
    _axis = XYZ.index(axis)
    if _axis == -1:
        raise ValueError("Invalid axis: {}.".format(axis))
    return _axis


def value_of_axis(arr, axis):
    return arr[:, axis2int(axis)]


def x_axis(arr):
    return value_of_axis(arr, 'x')


def y_axis(arr):
    return value_of_axis(arr, 'y')


def z_axis(arr):
    return value_of_axis(arr, 'z')


def compute_sigma2_factor(lors: np.ndarray):
    """
    compute the sigma2 factors of tor kernel
    """
    p_diff = p1(lors) - p0(lors)
    p_dis2 = np.sum(np.square(p_diff), 1)

    def square_sum_x_y(arr):
        return x_axis(arr)**2 + y_axis(arr)**2

    dsq = square_sum_x_y(p_diff)
    Rsq = square_sum_x_y(p0(lors)) + square_sum_x_y(p1(lors))
    sigma2_factor = dsq**2 / Rsq / p_dis2 / 2
    lors = np.hstack([lors, np.expand_dims(sigma2_factor, 1)])
    return lors


def partition_lors(lors: np.ndarray):
    """
    patition the input lors into three np.array according to the dominant direction.

    - `lors`: numpy.ndarry with shape [nb_events, nb_columns].
      Column order is ['x0', 'y0', 'z0', 'x1', 'y1', 'z1', ...]
    """
    nb_columns = lors.shape[1]
    diff = np.abs(p1(lors) - p0(lors))
    x_d, y_d, z_d = x_axis(diff), y_axis(diff), z_axis(diff)
    xlors = lors[np.array([np.intersect1d(np.where(x_d >= y_d),
                                          np.where(x_d >= z_d))])].reshape((-1, nb_columns))
    ylors = lors[np.array([np.intersect1d(np.where(y_d > x_d),
                                          np.where(y_d >= z_d))])].reshape((-1, nb_columns))
    zlors = lors[np.array([np.intersect1d(np.where(z_d > x_d),
                                          np.where(z_d > y_d))])].reshape((-1, nb_columns))
    return xlors, ylors, zlors

# exchange the start point and end point if the


def swap_points_order(lors: np.ndarray, axis: str):
    """
    Swap points order (0..3) <-> (3..6) to garantee main direction of p1 - p0 is postive.

    lors is an 2D array whose shape is [n, 6]
    Two points are indicated in the order of (x1, y1, z1, x2, y2, z2).
    This function garantees the point2 is large than the point1 in main axis(int (0, 1, 2) for (x, y, z))
    """

    d = value_of_axis(p0(lors), axis) - value_of_axis(p1(lors), axis)
    positive = lors[d >= 0]
    negative = lors[d < 0]
    to_hstack = [p1(negative), p0(negative)]
    if lors.shape[1] > 7:
        to_hstack += [0 - np.expand_dims(t_dis(negative), 1), extra(negative)]
    else:
        to_hstack += [np.expand_dims(t_dis(negative), 1)]
    return np.vstack((positive, np.hstack(to_hstack)))


def cut_lors(lors: np.ndarray, limit: np.float32):
    """
    form of an individul lor: (x1, y1, z1, x2, y2, z2, t_dis)
    cut the lors according to the tof information.
    this operation reduce the lor length due to the tof kernel
    return a lor: (x1, y1, z1, x2, y2, z2, xc, yc, zc)
    """
    point0 = p0(lors)
    point1 = p1(lors)
    p_diff = point1 - point0
    p_dis = np.sqrt(np.sum(np.square(p_diff), 1))
    dcos = np.array([value_of_axis(p_diff, a) / p_dis for a in XYZ]).T
    ratio = np.array(0.5 - (t_dis(lors) / p_dis))
    lor_center = np.array([ratio * value_of_axis(p_diff, a)
                           for a in XYZ]).T + point0

    # cut the lors
    dcs = np.sqrt(np.sum(np.square(lor_center - point0), 1))

    index = dcs > limit

    point0[index] = lor_center[index] - limit * dcos[index, :].reshape((-1, 3))

    # dce = np.sqrt(np.sum(np.square(lor_center - point1), 1))

    point1[index] = lor_center[index] + limit * dcos[index, :].reshape((-1, 3))
    return np.hstack((np.array(point0),
                      np.array(point1),
                      np.array(lor_center),
                      np.array(extra(lors))))


def lors2lors3(lors: np.ndarray, limit):
    """
    Partition lors to three ndarray of lors with x, y and z as main direction, respectively.
    """
    # TODO: Add unit test for this
    lors = compute_sigma2_factor(lors)
    xlors, ylors, zlors = partition_lors(lors)
    lors3 = {'x': xlors, 'y': ylors, 'z': zlors}
    lors3 = {a: swap_points_order(lors3[a], a) for a in XYZ}
    lors3 = {a: cut_lors(lors3[a], limit) for a in XYZ}
    lors3['x'] = lors3['x'][:, [1, 2, 0, 4, 5, 3, 7, 8, 6, 9]]
    lors3['y'] = lors3['y'][:, [0, 2, 1, 3, 5, 4, 6, 8, 7, 9]]
    return lors3


def calculate_limit(spec):
    return spec.tof.tof_res * spec.tor.c_factor / spec.tor.gaussian_factor * 3


# def update_specifics(recon_sepc, dist_spec):
#     """
#     Update reconsctruction specifics with new xlor files and splitted ranges.
#     """
#     nb_workers = self.nb_workers()
#     nb_subsets = self.osem_info.nb_subsets
#     self.tof_sigma2 = limit * limit / 9
#     self.tof_bin = self.tof_info.tof_bin * self.c_factor
#     lors_steps = {a: l.shape[0] // (nb_workers * nb_subsets)
#                   for a, l in lors3.items()}
#     lors_shapes = {a: [lors_steps[a], l.shape[1]] for a, l in lors3.items()}
#     self.lors_info = LorsInfo(
#         lors_files,
#         lors_shapes,
#         lors_steps,
#         None
#     )


def load_raw_lors(path_file: str, path_dataset=None):
    logging.debug('Raw lors path {}.'.format(path_file))
    if path_dataset is None:
        path_dataset = 'lors'
    p = Path(path_file)
    if p.suffix == '.h5':
        from dxl.data import ArrayNDHDF5
        d = ArrayNDHDF5(p, path_dataset)
        return d.data()
    elif p.suffix == '.npy':
        return np.load(p)
    elif p.suffix == '.npz':
        from dal.data.io import load_npz
        return load_npz(p)[path_dataset]


# class PreprocessSpec:
#     def __init__(self, config):
#         self.tof = ToFSpec(config['tof'])
#         self.lors = LoRsToRSpec(config['lors'])

#     def tor_recon_filename(self):
#         p = Path(recon_sepc.work_directory) / 'tor_recon.h5'
#         return str(p)


def process(spec, output=None):
    """
    Main function of preprocessing input data and config for TOR model reconstruction.

    - `recon_spec` reconstruction specifics
    - `dist_spec` distribute specifics
    - `output` output config file, if is not None, new specific will dumped to the .json file.
    """
    lors3 = lors2lors3(load_raw_lors(spec.tor.preprocess_lors.path_file,
                                     spec.tor.preprocess_lors.path_dataset),
                       calculate_limit(spec))
    print('seperate done')
    for k in lors3:
        print(k, lors3[k].shape)
    save_h5(spec.lors.path_file, lors3, spec.lors.path_dataset)
    if output is not None:
        raise NotImplementedError(
            "Dump of new specific it not implemented yet.")
