import numpy as np
from typing import Tuple, Callable, Dict, Any
import tensorflow as tf

from abc import ABCMeta, abstractclassmethod


class CompileAble(metaclass=ABCMeta):
    @abstractclassmethod
    def compile(self):
        pass

class Tensor(CompileAble):
    """
    对数据的进一步装箱，目的：

    - 统一数据单元接口
    - 构造没有实际内存，但是可以提供shape等属性的对象 （同时用于tf和更高层）
    - 提供外部信息空间
    - name等用于编译到tensorflow计算图的属性 （仅用于tf）
    - Inmutable 全部都是常量变量


    Notes:

    - np.ndarry 和 tf.Tensor 都是裸多维数组， 没有数组内部数据的解析方式的信息，
      Tensor是数据的实际内容(self.data)和它们的解析方式(实际表现为对它们的操作)的封装。
    """
    required_interface = ()

    def __init__(self, data, shape=None, name=None, info=None):
        self.data: np.ndarray or tf.Tensor or None
        self.shape: tuple
        self.name: str
        self.host: 'Host'
        self.info: Dict[str, Any]  # JSON Serializable info
        self.export_interface = ()
        # Check interfaces

    def __getattr__(self, attr):
        for i in self._export_interface:
            if hasattr(i, attr):
                return getattr(i, attr)

    def copy_to(self, host) -> 'Tensor':
        self.data : tf.Tensor
        with tf.device(host):
            data_new: tf.Tensor = tf.constant(self.data)
            return Tensor(data_enw, host)



class TensorFromH5(Tensor):

    def __init__(self, fliename, lazy=True, key=None):
        pass

    def compile(self):
        if lazy:
            op: 'TensorFlow load local op (by py_func warp)'
            return op
        else:
            data: np.numpy  # Load data from h5
            return tf.constant(data, self.name)


class Interface(metaclass=ABCMeta):
    """
    Interface class, indicating there is no Tensor member in this class
    """
    pass


class OperationWithTFSupport:
    tf_op = None


class Operation(metaclass=ABCMeta):
    pass


class Multidimensional(Interface):
    def dim(self) -> Callable[[Tensor], int] or int:
        pass


class Discretization(Interface):
    def resample_to(self, new_discritization: 'Discritization') -> Callable[[Tensor], Tensor]:
        raise NotImplementedError


class Cartesian(Discretization, Multidimensional):
    def dim(self) -> Callable[[Tensor], int]:
        def dim_(t: Tensor):
            if isinstance(t.data, np.ndarray):
                return t.ndim


class Blob(Discritization, Multidimensional):
    pass


class Blob3D:
    def dim(self):
        return lambda t: 3


class DiscreteVolume(Tensor):
    required_discretization = ()

    def __init__(self, data=None, discretization=None):
        super().__init__(data)
        for d in self.required_discretization():
            if not isinstance(discretization, d):
                raise TypeError(
                    "Required discretization {} not satisfied for {}.".format(d, __class__))
        self.discretization: Discritization

    def resample_to(self, new_discritization: Discretization) -> 'DiscreteVolume':
        return self.discretization.resample_to(new_discritization)(self.data)

    @property
    def dim(self):
        return self.discretization(self.data)

    def __getattr__(self, attr):
        return getattr(self.discretization, attr)(self.data)


"""
Example usage

``` python

class PETImageCartesian3D(DiscreteVolume):
    def __init__(self, data: np.ndarray):
        super().__init__(data, Cartesian3D())
    


```
"""
