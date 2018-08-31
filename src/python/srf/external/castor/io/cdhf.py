from doufo import dataclass
import numpy as np
from doufo import List

@dataclass
class CrystalPair():
    id1: int
    id2: int


@dataclass
class DataHeader():
    data_file_name: str
    scanner_name: str
    nb_events: int
    data_mode: str = "list-mode" # can be "list-mode", "histogram", or "normalization" 
    data_type: str = "PET"
    start_time: float
    duration: float

    max_nb_lines_per_event: int = 1
    max_axial_difference: float = -1.0
    calibration_factor: float = 1.0
    attenuation_correction_flag: bool = False
    normalization_correction_flag: bool = False
    scatter_correction_flag: bool = False
    random_correction_flag: bool = False
    isotope: str
    tof_info_flag: bool = False
    tof_resolution: float
    nb_tof_bins: int
    tof_bin_size: float
    tof_range: float    

@dataclass
class DataHistogramEvent():
    time: int
    attenuation_factor: float
    unnormalization_random_rate: float
    normalization_factor: float
    amount_in_histogram_bin:float
    unnormalization_scatter_rate: float
    nb_crystal_pairs: int
    crystal_pairs:List[CrystalPair]

@dataclass
class DataListModeEvent():
    time: int
    atteunation_factor: float
    unnormalization_scatter_rate: float
    unnormalization_random_rate: float
    normalization_factor: float
    tof_diff_time: float
    nb_crystal_pairs: int
    crystal_pairs:List[CrystalPair]

@dataclass
class DataNormalizationEvent():
    attenuation_factor: float
    normalization_factor: float
    nb_crystal_pairs: int
    crystal_pairs:List[CrystalPair]

