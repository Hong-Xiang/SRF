from typing import Dict, Any
from srf.data import ListModeData, PositionEvent, LoR, ListModeDataWithoutTOF, ListModeDataSplitWithoutTOF
from doufo import x
from doufo.tensor import backends, shape
import numpy as np
from doufo.tensor import const

__all__ = ['from_tensor', 'to_tensors', 'create_listmode_data']


def is_position_event(lors: ListModeData) -> bool:
    # TODO a better implementation is needed
    return isinstance(lors[0].fst, PositionEvent)


DEFAULT_WEIGHT_COLUMN = 6


def to_tensors(data: ListModeData) -> Dict[str, Any]:
    nb_lors = len(data)
    result = {}
    result['fst'] = fetch_to_np_array(data, x.fst.position)
    result['snd'] = fetch_to_np_array(data, x.snd.position)
    result['weight'] = fetch_to_np_array(data, x.weight)
    if is_with_tof(data):
        result['tof'] = fetch_to_np_array(data, x.tof)
    return result


def from_tensor(data, columns=None):
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


from doufo import tagfunc


@tagfunc(nargs=1, nouts=1)
def create_listmode_data(raw_data):
    raise NotImplementedError("No default implementation for create_listmode_data.")


@create_listmode_data.register(ListModeDataWithoutTOF)
def _(raw_data):
    return ListModeDataWithoutTOF(const[backends.TensorFlowBackend](raw_data.astype(np.float32), name='lors'),
                                  const[backends.TensorFlowBackend](np.ones([shape(raw_data)[0], 1], dtype=np.float32),
                                                                    name='values'))


@create_listmode_data.register(ListModeDataSplitWithoutTOF)
def _(raw_data):
    from srf.preprocess.function.on_tor_lors import Axis
    lors = {a: const[backends.TensorFlowBackend](raw_data[a].astype(np.float32), name=f'lors_{a.value}')
            for a in Axis}
    values = {a: np.ones([raw_data[a].shape[0], 1], dtype=np.float32) for a in Axis}
    values = {a: const[backends.TensorFlowBackend](v, name=f"lors_value_{a}") for a, v in values.items()}
    return ListModeDataSplitWithoutTOF(ListModeDataWithoutTOF(lors[Axis.x], values[Axis.x]),
                                       ListModeDataWithoutTOF(lors[Axis.y], values[Axis.y]),
                                       ListModeDataWithoutTOF(lors[Axis.z], values[Axis.z]))
