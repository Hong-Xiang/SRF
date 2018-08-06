from doufo import List, dataclass, Pair
from doufo.tensor import Tensor
from typing import Optional


__all__ = ['Event', 'DetectorIdEvent', 'PositionEvent', 'LoR', 'ListModeDataXYZSplitted']


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



class ListModeDataXYZSplitted:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

