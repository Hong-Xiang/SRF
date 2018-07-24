from typing import Dict
from srf.data import ListModeData, Tensor, PositionEvent
import numpy as np


def is_position_event(lors: ListModeData) -> bool:
    return isinstance(lors[0].fst, PositionEvent)


def to_tensors(data: ListModeData) -> Dict[str, Tensor]:
    nb_lors = len(data)
    result = np.zeros([nb_lors, ])
