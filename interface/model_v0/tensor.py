import numpy as np
import tensorflow as tf
from typing import Tuple, Callable, Dict, Any
from abc import ABCMeta, abstractclassmethod

from interface import PhysicsCartesian


class CompileAble(metaclass=ABCMeta):
    """
    提供一个编译数据为计算图的接口
    """
    @abstractclassmethod
    def copmile(self):
        pass

class Tensor(CompileAble):
    """
    对数据的进一步装箱，目的：
    - 统一数据单元接口
    - 构造没有实际内存，但是可以提供raw_shape等属性的对象 （同时用于tf和更高层）
    - 提供外部信息空间
    - name等用于编译到tensorflow计算图的属性 （仅用于tf）
    - Inmutable 全部都是常量变量
    Notes:
    - np.ndarry 和 tf.Tensor 都是裸多维数组， 没有数组内部数据的解析方式的信息，
      Tensor是数据的实际内容(self.data)和它们的解析方式(实际表现为对它们的操作)的封装。
    """
    reqiured_interface=()
    def __init__(self, data, raw_shape=None, name=None, info=None):
        self.data: np.ndarray or tf.Tensor or None
        self.raw_shape: tuple
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
            return Tensor(data_new, host)
    
    @abstractclassmethod
    def save(self, file_name):
        """
        将Tensor数据储存到文件中。
        """
        pass
    @abstractclassmethod
    def load(self,file_name):
        """
        从文件中加载一个Tensor的数据。
        """
        pass

class TensorFromH5(Tensor):
    def __init__(self, fliename, key=None):
        pass

    def compile(self):
        data: np.ndarray  # Load data from h5
        return tf.constant(data, self.name)

class VectorLowDim(Tensor):
    dim = None
    pass


class Vector2(VectorLowDim):
    dim = 2

    def __init__(self, x, y):
        pass

    @property
    def x(self):
        pass

    @property
    def y(self):
        pass


class Vector3(VectorLowDim):
    """
    一个三维空间中的向量
    """
    dim = 3

    def __init__(self, x, y, z):
        pass

    @property
    def x(self):
        pass

    @property
    def y(self):
        pass

    @property
    def z(self):
        pass



    





    
