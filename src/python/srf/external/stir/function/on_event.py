from srf.data import PositionEvent, DetectorIdEvent, PETCylindricalScanner
import numpy as np
from doufo.tensor import norm,Vector

__all__ = ['position2detectorid']


def position2detectorid(scanner: PETCylindricalScanner, event: PositionEvent) -> DetectorIdEvent:
    return DetectorIdEvent(ring_id(scanner, event.position),
                           block_id(scanner, event.position),
                           crystal_id(scanner, event.position))


def block_id(scanner, position):
    return int(theta(scanner, position) / block_angle(scanner) + 0.5) % scanner.nb_blocks_per_ring


def theta(scanner, position):
    result = np.math.acos(Vector(position).x / norm(p_xy(position)))
    if Vector(position).y < 0:
        result = 2 * np.pi - result
    return result


def p_xy(position):
    return position[:2]


def block_angle(scanner):
    return 2.0 * np.pi / scanner.nb_blocks_per_ring


def ring_id(scanner, position):
    return int((Vector(position).z / scanner.axial_length + 0.5) * scanner.nb_rings*scanner.blocks[0].grid[2])


def crystal_id(scanner, position):
    bid = block_id(scanner, position)
    shift = ((p_xy(position) - center(scanner, bid))) @ normal(scanner, bid)
    result = int((shift + scanner.block_width / 2) / scanner.crystal_width)
    # FIXME handle of odd nb_detectors_per_block
    previous = bid * scanner.nb_detectors_per_block
    return result + previous


def direction(scanner, block_id_):
    phi = block_id_ * block_angle(scanner)
    return np.array([np.math.cos(phi), np.math.sin(phi)])


def normal(scanner, block_id_):
    d = direction(scanner, block_id_)
    return np.array([-d[1], d[0]])


def center(scanner, block_id_):
    return direction(scanner, block_id_) * scanner.inner_radius
