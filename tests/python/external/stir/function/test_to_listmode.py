from srf.data import ListModeData, LoR, PositionEvent
from srf.external.stir.function import ndarray2listmode
from dxl.data.tensor import Point


def test_ndarray2listmode(l2sdata):
    # HACK hard-coded test data, need to figure out elegant way of data-driven test.
    listmode = ndarray2listmode(l2sdata['input'])
    assert len(listmode) == 100
    assert isinstance(listmode[0].fst, PositionEvent)
    assert isinstance(listmode[0].snd, PositionEvent)
    assert listmode[0].fst.position == Point(l2sdata['input'][0, :3])
    assert listmode[0].snd.position == Point(l2sdata['input'][0, 3:6])
