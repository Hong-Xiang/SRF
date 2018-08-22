from doufo import Function, func, dataclass
from doufo.tensor import abs_, norm
from doufo.collections import DataArray
from srf.data.listmode import LOR

from enum import Enum
import numpy as np


class Axis(Enum):
    x = 0
    y = 1
    z = 2


@dataclass
class Direction3:
    x: float
    y: float
    z: float

    def fmap(self, f):
        return Direction3(*f([x, y, z]))


class HitOfIndex(Function):
    def __init__(self, index):
        self.index = index

    def __call__(self, x):
        return x[:, self.index * 3:(self.index + 1) * 3]


@func
def hit_of_index(index, arr):
    return arr[:, index*3:(index+1)*3]


@func
def direction(lors):
    return Direction3(lors.snd.x - lors.fst.x,
                      lors.snd.y - lors.fst.y,
                      lors.snd.z - lors.fst.z)


class SliceByAxis(Function):
    def __init__(self, axis: Axis):
        self.axis = axis

    def __call__(self, lors3):
        return lors3[:, self.axis.value]


def dominentBy(axis: Axis, x_d, y_d, z_d):
    if axis == Axis.x:
        conds = [np.where(x_d >= y_d), np.where(x_d >= z_d)]
    if axis == Axis.y:
        conds = [np.where(y_d > x_d), np.where(y_d >= z_d)]
    if axis == Axis.z:
        conds = [np.where(z_d > x_d), np.where(z_d > y_d)]
    return np.array([np.intersect1d(*conds)])


@func
def dominent_by(axis: Axis, d: Direction3):
    if axis == Axis.X:
        return d.x >= d.y & d.x >= d.z
    if axis == Axis.Y:
        return d.y > d.x & d.y >= d.z
    if axis == Axis.Z:
        return d.z > d.x & d.z > d.y

@func
def partition(axis: Axis, lors):
    d = direction(lor).fmap(abs_)
    index = dominent_by(axis, d)
    return lors[index]


class Partition(Function):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, lors):
        diff = abs_(direction(lors))
        directions = [SliceByAxis(a)(diff) for a in Axis]
        return lors[dominentBy(self.axis, *directions)].reshape((-1, lors.shape[1]))


def tof(lors):
    """get the tof column
    """
    return lors[:, 6]


def extra(lors):
    """get the information of lors beyond the position.
    """
    return lors[:, 7:]


def arr_norm(arr):
    return np.sqrt(np.sum(np.square(arr), 1))


class SquareOf(Function):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, arr):
        return SliceByAxis(self.axis)(arr)**2


def sigma2_factor(lors):
    """compute the variable tor kernel sigma factor square.

    The sigma factor is used to provide a shift-variant tor kernel.
    A single lor 

    """
    p_diff = direction(lors)
    p_dis2 = np.sum(np.square(p_diff), 1)
    p0, p1 = HitOfIndex(0)(lors), HitOfIndex(1)(lors)

    def square_sum(arr):
        return SquareOf(Axis.x)(arr) + SquareOf(Axis.y)(arr)
    dsq = square_sum(p_diff)
    Rsq = square_sum(p0) + square_sum(p1)
    return dsq**2 / Rsq / p_dis2 / 2


def partition3(lors):
    """partition the lors into three groups according to the dominent direction. 
    """
    return Partition(Axis.x)(lors), Partition(Axis.y)(lors), Partition(Axis.z)(lors)


class SwapPointsOrder(Function):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, lors):
        d = (direction >> SliceByAxis(self.axis))(lors)
        positive = lors[d >= 0]
        negative = lors[d < 0]
        to_hstack = [HitOfIndex(1)(negative), HitOfIndex(0)(negative)]

        if lors.shape[1] > 7:
            to_hstack += [0 -
                          np.expand_dims(tof(negative), 1), extra(negative)]
        else:
            to_hstack += [np.expand_dims(tof(negative), 1)]
        return np.vstack((positive, np.hstack(to_hstack)))


class CutLoRs(Function):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, lors):
        p_diff = direction(lors)
        p_dis = arr_norm(p_diff)

        dcos = np.array(
            [p_diff[:, a.value] / p_dis for a in Axis]).T.reshape((-1, 3))

        ratio = np.array(0.5 - (tof(lors) / p_dis))

        p0 = HitOfIndex(0)(lors)
        p1 = HitOfIndex(1)(lors)
        lor_center = np.array([ratio * p_diff[:, a.value]
                               for a in Axis]).T + p0

        dcs = arr_norm(lor_center - p0)
        index = dcs > self.limit

        p0[index] = lor_center[index] - self.limit * dcos[index, :]
        p1[index] = lor_center[index] + self.limit * dcos[index, :]

        return np.hstack((np.array(p0),
                          np.array(p1),
                          np.array(lor_center),
                          np.array(extra(lors))
                          ))


def compute_sigma2_factor_and_append(lors):
    return np.hstack([lors, np.expand_dims(sigma2_factor(lors), 1)])


@function
def func_on(a: Axis):
    return Partition(a) >> SwapPointsOrder(a)

# on io.py


def map_process(lors):
    lors = compute_sigma2_factor_and_append(lors)
    lors3 = {a: func_on(a)(lors) for a in Axis}
    lors3[Axis.x][:, 0:6] = lors3[Axis.x][:, [1, 2, 0, 4, 5, 3]]
    lors3[Axis.y][:, 0:6] = lors3[Axis.y][:, [0, 2, 1, 3, 5, 4]]
    return lors3


def recon_process(lors, limit):
    lors = compute_sigma2_factor_and_append(lors)
    lors3 = {a: func_on(a)(lors) for a in Axis}
    lors3 = {a: CutLoRs(limit)(lors3[a]) for a in Axis}
    lors3[Axis.x][:, 0:9] = lors3[Axis.x][:, [1, 2, 0, 4, 5, 3, 7, 8, 6]]
    lors3[Axis.y][:, 0:9] = lors3[Axis.y][:, [0, 2, 1, 3, 5, 4, 6, 8, 7]]
    return lors3
