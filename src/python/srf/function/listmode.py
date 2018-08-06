from typing import Dict
from srf.data import ListModeData, Tensor, PositionEvent, LoR
from dxl.function import x
import numpy as np

__all__ = ['from_tensor', 'to_tensors']


def is_position_event(lors: ListModeData) -> bool:
    # TODO a better implementation is needed
    return isinstance(lors[0].fst, PositionEvent)


DEFAULT_WEIGHT_COLUMN = 6


def to_tensors(data: ListModeData) -> Dict[str, Tensor]:
    nb_lors = len(data)
    result = {}
    result['fst'] = fetch_to_np_array(data, x.fst.position)
    result['snd'] = fetch_to_np_array(data, x.snd.position)
    result['weight'] = fetch_to_np_array(data, x.weight)
    if is_with_tof(data):
        result['tof'] = fetch_to_np_array(data, x.tof)
    return result


def from_tensor(data: Tensor, columns=None):
    columns = auto_complete_columns(data, columns)
    return ListModeData([
        LoR(PositionEvent(data[i, columns['fst']]),
            PositionEvent(data[i, columns['snd']]),
            weight=maybe_weight(data, i, columns))
        for i in range(data.shape[0])])


def auto_complete_columns(data, columns):
    if columns is None:
        columns = {'fst': slice(0, 3), 'snd': slice(3, 6)}
        if with_weight(data):
            columns['weight'] = DEFAULT_WEIGHT_COLUMN
    return columns


def with_weight(data):
    return data.shape[1] >= DEFAULT_WEIGHT_COLUMN + 1


def maybe_weight(data, i, columns):
    if 'weight' in columns:
        return data[i, columns['weight']]
    else:
        return None


def fetch_to_np_array(data, func):
    return np.array(data.fmap(func))


def is_with_tof(data) -> bool:
    return False
