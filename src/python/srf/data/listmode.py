from dxl.data import List, DataClass, Pair

__all__ = ['Event', 'DetectorIdEvent', 'PositionEvent', 'LoR', 'ListModeData']


class Event(DataClass):
    pass


class DetectorIdEvent(Event):
    __slots__ = ('id_ring', 'id_block', 'id_crystal')


class PositionEvent(Event):
    __slots__ = ('position', )


LoR = Pair[Event, Event]

ListModeData = List[LoR]
