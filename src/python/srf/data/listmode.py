from dxl.data import List, DataClass, Pair

__all__ = ['Event', 'DetectorIdEvent', 'PositionEvent', 'LoR', 'ListModeData']


class Event(DataClass):
    pass


class DetectorIdEvent(Event):
    __slots__ = ('id_ring', 'id_block', 'id_crystal')


class PositionEvent(Event):
    __slots__ = ('position', )


class LoR(Pair[Event, Event]):
    def __init__(self, e0, e1, weight=1.0, tof=None):
        super().__init__(e0, e1)
        self.weight = weight
        self.tof = tof

    def flip(self):
        return LoR(self.snd, self.fst, self.weight, -self.tof)

    def __repr__(self):
        return f"<LoR({self.fst}, {self.snd}, {self.weight}, {self.tof})>"


ListModeData = List[LoR]


from dxl.data.tensor import Tensor


class LoRArray(Pair[Tensor, Tensor]):
    def __init__(self, h0, h1, weight=None, tof=None):
        super().__init__(h0, h1)
        if weight is None:
            weight = Tensor(np.ones([h0.shape[0]]))
        self.weight = weight
        self.tof = tof


ListModeDataXYZSplitted = None

# List[LoR]
# LoR[Event[Tensor], Event[Tensor]]

from dxl.data import DataArray


class ListModeDataV2(DataArray):
    pass
