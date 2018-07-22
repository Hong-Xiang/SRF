from dxl.data import List, Pair
from dxl.data.tensor import Point
from dxl.function import x
from srf.data import Event, LoR, ListModeData, PETSinogram3D, PETCylindricalScanner
import numpy as np


def listmode2sinogram(scanner: PETCylindricalScanner, listmode_data: ListModeData) -> PETSinogram3D:
    return accumulating2sinogram(scanner, listmode_data.fmap(rework_indices))


def rework_indices(scanner: PETCylindricalScanner, lor: LoR):
    return fix_ring_index(lor.fmap2(lambda e: fix_crystal_id(scanner, e)),
                          scanner.nb_detectors)


def fix_crystal_id(scanner, event):
    fixed_id = event.id_crystal + scanner.nb_detectors // 4 % scanner.nb_detectors
    return event.replace(id_crystal=fixed_id)


def fix_ring_index(lor, nb_detectors):
    crystal_centers = lor.fmap2(
        lambda e: center_of_crystal(e.id_crystal, nb_detectors))
    ring_ids, crystal_ids = lor.fmap(x.id_ring), lor.fmap(x.id_crystal)
    if is_need_swap_crystal_id(crystal_centers):
        ring_ids = ring_ids.flip()
    if is_need_swap_rind_id(crystal_centers):
        crystal_ids = crystal_ids.flip()
    return LoR(Event(ring_ids.fst, crystal_ids.fst),
               Event(ring_ids.snd, crystal_ids.snd))


def center_of_crystal(crystal_id, nb_detectors):
    x = np.sin((0.5 + crystal_id) * (2 * np.pi) / nb_detectors)
    y = np.cos((0.5 + crystal_id) * (2 * np.pi) / nb_detectors)
    return Point([x, y])


def is_need_swap_rind_id(ps: Pair[Point, Point]) -> bool:
    if ps.fst.x > ps.snd.x:
        return True
    if ps.fst.x == ps.snd.x and ps.fst.y < ps.snd.y:
        return True
    return False


def is_need_swap_crystal_id(ps: Pair[Point, Point]) -> bool:
    if ps.fst.x < ps.snd.x:
        return True
    if ps.fst.x == ps.snd.x and ps.fst.y > ps.snd.y:
        return True
    return False


def accumulating2sinogram(scanner, lors: ListModeData):
    nb_views_, nb_sinograms_ = nb_views(scanner), nb_sinograms(scanner)
    result = np.zeros([nb_views_, nb_views_, nb_sinograms_])
    for lor in lors:
        ring_ids = lors.fmap2(x.id_ring)
        id_bin_ = id_bin(scanner, ring_ids)
        if id_bin_ > 0 and id_bin_ < nb_views_:
            result[id_sinogram(scanner, ring_ids), id_bin_,
                   id_view(scanner, ring_ids)] += 1
    return PETSinogram3D(result)


def nb_views(scanner):
    return scanner.nb_detectors // 2


def nb_sinograms(scanner):
    return scanner.nb_rings * scanner.nb_rings


def id_sinogram(scanner, ring_ids):
    delta_z = ring_ids.snd - ring_ids.fst
    result = (ring_ids.fst + ring_ids.snd - abs(delta_z)) // 2 + \
        (scanner.nb_ring if delta_z != 0 else 0)
    for i in range(1, abs(delta_z)):
        result += 2 * (scanner.nb_rings - i)
    if(delta_z < 0):
        result += scanner.nb_ring - abs(delta_z)
    return result


def id_view(scanner, ring_ids):
    return (int(sum(ring_ids) + scanner.nb_detectors // 2 + 1) / 2) % scanner.nb_detectors // 2


def id_bin(scanner, ring_ids):
    def diff(id_):
        return min(abs(id_ - id_view), abs(id_ - (id_view + scanner.nb_detectors)))
    diffs = ring_ids.fmap(diff)
    if (abs(diffs.fst) < abs(diffs.snd)):
        sigma = ring_ids.fst - ring_ids.snd
    else:
        sigma = ring_ids.snd - ring_ids.fst

    if (sigma < 0):
        sigma += scanner.nb_detectors
    return sigma + (nb_views) / 2 - scanner.nb_detectors / 2
