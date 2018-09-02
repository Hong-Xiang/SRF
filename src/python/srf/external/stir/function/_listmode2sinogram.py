from doufo import List, Pair, x
from dxl.shape.data import Point
from srf.data import DetectorIdEvent, LoR,PETSinogram3D, PETCylindricalScanner, PositionEvent, DetectorIdEvent
from .on_event import position2detectorid
import numpy as np
from functools import partial

__all__ = ['listmode2sinogram']


def listmode2sinogram(scanner: PETCylindricalScanner, listmode_data: List[LoR]) -> PETSinogram3D:    
    listmode_data = ensure_detectorid_event(scanner, listmode_data)
    data_with_fixed_ids = listmode_data.fmap(partial(rework_indices, scanner))
    return accumulating2sinogram(scanner, data_with_fixed_ids)


def ensure_detectorid_event(scanner, listmode_data: List[LoR]) -> List[LoR]:
    if len(listmode_data) == 0:
        return listmode_data
    return listmode_data.fmap(lambda l: l.fmap2(partial(maybe_convert_to_detectorid_event, scanner)))


def maybe_convert_to_detectorid_event(scanner, e):
    if isinstance(e, PositionEvent):
        return position2detectorid(scanner, e)
    else:
        return e


def rework_indices(scanner: PETCylindricalScanner, lor: LoR):
    return fix_ring_index(lor.fmap2(lambda e: fix_crystal_id(scanner, e)),
                          scanner.nb_detectors_per_ring)


def fix_crystal_id(scanner, event: DetectorIdEvent)->int:
    fixed_id = ((event.id_crystal + scanner.nb_detectors_per_ring // 4)
                % scanner.nb_detectors_per_ring)
    return event.replace(id_crystal=fixed_id)


def fix_ring_index(lor: LoR, nb_detectors: int) -> LoR:
    direction = lor.fmap2(
        lambda e: center_of_crystal(e.id_crystal, nb_detectors))
    ring_ids, crystal_ids = lor.fmap2(x.id_ring), lor.fmap2(x.id_crystal)
    if is_need_swap_ring_id(direction):
        ring_ids = ring_ids.flip()
    if is_need_swap_crystal_id(direction):
        crystal_ids = crystal_ids.flip()
    return LoR(DetectorIdEvent(ring_ids.fst, None, crystal_ids.fst),
               DetectorIdEvent(ring_ids.snd, None, crystal_ids.snd))


def center_of_crystal(crystal_id: int, nb_detectors: int) -> Point:
    x = np.sin((0.5 + crystal_id) * (2 * np.pi) / nb_detectors)
    y = np.cos((0.5 + crystal_id) * (2 * np.pi) / nb_detectors)
    return Point([x, y])


def is_need_swap_ring_id(ps: Pair) -> bool:
    if ps.fst.x > ps.snd.x:
        return True
    if ps.fst.x == ps.snd.x and ps.fst.y < ps.snd.y:
        return True
    return False


def is_need_swap_crystal_id(ps: Pair) -> bool:
    if ps.fst.x < ps.snd.x:
        return True
    if ps.fst.x == ps.snd.x and ps.fst.y > ps.snd.y:
        return True
    return False


def accumulating2sinogram(scanner, lors: List[LoR]) -> PETSinogram3D:
    nb_views_, nb_sinograms_ = nb_views(scanner), nb_sinograms(scanner)
    result = np.zeros([nb_sinograms_, nb_views_, nb_views_])
    for lor in lors:
        ring_ids, crystal_ids = lor.fmap2(x.id_ring), lor.fmap2(x.id_crystal)
        id_bin_ = id_bin(scanner, crystal_ids)
        if id_bin_ >= 0 and id_bin_ < nb_views_:
            result[id_sinogram(scanner, ring_ids),
                   id_view(scanner, crystal_ids),
                   id_bin_] += 1
    return PETSinogram3D(result)


def nb_views(scanner) -> int:
    return scanner.nb_detectors_per_ring // 2


def nb_sinograms(scanner) -> int:
    return scanner.nb_rings * scanner.nb_rings *scanner.blocks[0].grid[2] *scanner.blocks[0].grid[2]


def id_sinogram(scanner, ring_ids: Pair) -> int:
    delta_z = ring_ids.snd - ring_ids.fst
    result = (ring_ids.fst + ring_ids.snd - abs(delta_z)) / 2.0
    if delta_z != 0:
        result += scanner.nb_rings *scanner.blocks[0].grid[2]
    for i in range(1, abs(delta_z)):
        result += 2 * (scanner.nb_rings *scanner.blocks[0].grid[2] - i)
    if(delta_z < 0):
        result += (scanner.nb_rings *scanner.blocks[0].grid[2] - abs(delta_z))
    return int(result)


def id_view(scanner, crystal_ids: Pair) -> int:
    half_dct = scanner.nb_detectors_per_ring // 2
    return (crystal_ids.fst + crystal_ids.snd + half_dct + 1) // 2 % half_dct


def id_bin(scanner, crystal_ids: Pair) -> int:
    crystal_ids = maybe_fliped_cystal_ids(scanner, crystal_ids)
    result = (crystal_ids.snd - crystal_ids.fst) % scanner.nb_detectors_per_ring
    result += nb_views(scanner) // 2 - scanner.nb_detectors_per_ring // 2
    return result


def maybe_fliped_cystal_ids(scanner, crystal_ids):
    diffs = crystal_ids.fmap2(partial(crystal_id_view_id_difference,
                                      id_view(scanner, crystal_ids),
                                      scanner.nb_detectors_per_ring))
    return crystal_ids.flip() if diffs.fst < diffs.snd else crystal_ids


def crystal_id_view_id_difference(view_id, nb_detectors, crystal_id):
    return min(abs(crystal_id - view_id),
               abs(crystal_id - (view_id + nb_detectors)))
