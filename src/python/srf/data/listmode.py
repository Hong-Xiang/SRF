from dxl.data import List, DataClass, Pair

__all__ = ['Event', 'DetectorIdEvent', 'PositionEvent', 'LoR', 'ListModeData']


class Event(DataClass):
    pass


class DetectorIdEvent(Event):
    __slots__ = ('id_ring', 'id_block', 'id_crystal')


class PositionEvent(Event):
    __slots__ = ('position', )


class LoR(Pair[Event, Event]):
    def __init__(self, e0, e1, weight=None, tof=None):
        super().__init__(e0, e1)
        self.weigth = weight
        self.tof = tof

    def flip(self):
        return LoR(self.snd, self.fst, self.weight, -self.tof)


ListModeData = List[LoR]
