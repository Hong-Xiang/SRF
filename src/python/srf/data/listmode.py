from doufo import List, dataclass, Pair
from doufo.tensor import Tensor
from doufo import converters
from typing import Optional
import abc

__all__ = ['Event', 'DetectorIdEvent', 'PositionEvent', 'LoR', 'ListModeDataSplit', 'ListModeData',
           'ListModeDataWithoutTOF', 'ListModeDataSplitWithoutTOF']


@dataclass
class Event:
    pass


@dataclass
class DetectorIdEvent:
    id_ring: int
    id_block: int
    id_crystal: int


@dataclass
class PositionEvent(Event):
    position: Tensor


@dataclass
class LoR(Pair):
    fst: Event
    snd: Event
    weight: float = 1.0
    tof: Optional[float] = None

    def flip(self):
        tof = None if self.tof is None else -self.tof
        return self.replace(fst=self.snd, snd=self.fst, tof=tof)


class AbstractListModeData(abc.ABC):
    @property
    @abc.abstractmethod
    def lors(self):
        pass

    @property
    @abc.abstractmethod
    def values(self):
        pass


class ListModeData(AbstractListModeData):
    def __init__(self, lors, values):
        self._lors = lors
        self._values = values

    @property
    def lors(self):
        return self._lors

    @property
    def values(self):
        return self._values


class ListModeDataWithoutTOF(AbstractListModeData):
    def __init__(self, lors, values):
        self._lors = lors
        self._values = values

    @property
    def lors(self):
        return self._lors

    @property
    def values(self):
        return self._values


# TODO Implementation

@converters.register(ListModeData, ListModeDataWithoutTOF)
def _(lors):
    raise NotImplementedError


class AbstractSplitListModeData(abc.ABC):
    @property
    @abc.abstractmethod
    def x(self):
        pass

    @property
    @abc.abstractmethod
    def y(self):
        pass

    @property
    @abc.abstractmethod
    def z(self):
        pass


class XYZGetterMixin:
    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def __getitem__(self, item):
        if item == 'x':
            return self.x
        if item == 'y':
            return self.y
        if item == 'z':
            return self.z
        raise KeyError(item)


class ListModeDataSplit(XYZGetterMixin, AbstractSplitListModeData):
    def __init__(self, x: ListModeData, y: ListModeData, z: ListModeData):
        self._x = x
        self._y = y
        self._z = z


class ListModeDataSplitWithoutTOF(XYZGetterMixin, AbstractSplitListModeData):
    def __init__(self, x: ListModeDataSplit, y: ListModeDataSplit, z: ListModeDataSplit):
        self._x = x
        self._y = y
        self._z = z
