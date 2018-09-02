from doufo import dataclass
import numpy as np
from doufo import List


__all__ = ["CrystalPair", "DataHeader", "DataHistogramEvent",
           "DataListModeEvent", "DataNormalizationEvent"]


@dataclass
class CrystalPair:
    id1: int
    id2: int


@dataclass
class DataHeader:
    data_file_name: str
    scanner_name: str
    nb_events: int

    start_time: float
    duration: float

    tof_resolution: float = -1.0
    nb_tof_bins: int = -1
    tof_bin_size: float = -1.0
    tof_range: float = 0.0

    data_mode: str = "list-mode"  # can be "list-mode", "histogram", or "normalization"
    data_type: str = "PET"
    max_nb_lines_per_event: int = 1
    max_axial_difference: float = -1.0
    calibration_factor: float = 1.0
    attenuation_correction_flag: bool = False
    normalization_correction_flag: bool = False
    scatter_correction_flag: bool = False
    random_correction_flag: bool = False
    isotope: str = "unknown"
    tof_info_flag: bool = False



@dataclass
class DataHistogramEvent:
    time: int
    attenuation_factor: float
    unnormalization_random_rate: float
    normalization_factor: float
    amount_in_histogram_bin: float
    unnormalization_scatter_rate: float
    nb_crystal_pairs: int
    crystal_pairs: List[CrystalPair]


@dataclass
class DataListModeEvent():
    time: int
    atteunation_factor: float
    unnormalization_scatter_rate: float
    unnormalization_random_rate: float
    normalization_factor: float
    tof_diff_time: float
    nb_crystal_pairs: int
    crystal_pairs: List[CrystalPair]


@dataclass
class DataNormalizationEvent:
    attenuation_factor: float
    normalization_factor: float
    nb_crystal_pairs: int
    crystal_pairs: List[CrystalPair]
