from doufo import dataclass
import numpy as np
from doufo import List


__all__ = ["CrystalPair", "DataHeader", "DataHistogramEvent",
           "DataHeaderScript", "DataListModeEvent", "DataNormalizationEvent"]


@dataclass
class CrystalPair:
    id1: int
    id2: int

class DataHeader:
    def __init__(self, config:dict):
        if not isinstance(config, dict):
            raise TypeError("The data header config type is not dict.")
        self.data_file_name =  config['data_file_name']
        self.scanner_name = config['scanner_name']
        self.nb_events= config['nb_events']
        self.start_time = config['start_time']
        self.duration = config['duration']
        self.data_mode = config['data_mode']
        self.data_type = config['data_type']
        self.max_axial_difference = config['max_axial_difference']
        self.max_nb_lines_per_event = config['max_nb_lines_per_event']
        self.calibration_factor = config['calibration_factor']
        self.isotope = config['isotope']
        self.attenuation_correction_flag = config['attenuation_correction_flag']
        self.normalization_correction_flag = config['normalization_correction_flag']
        self.scatter_coorection_flag = config['scatter_correction_flag']
        self.random_correction_flag = config['random_correction_flag']
        self.tof_info_flag = config['tof_info_flag']
        self.nb_tof_bins = config['nb_tof_bins']
        self.tof_bin_size = config['tof_bin_size']
        self.tof_range = config['tof_range']

class DataHeaderScript:
    template = 'cdh.j2'

    def __init__(self, spec):
        self.spec = spec

    def render(self, template) -> str:
        return template.render(header=self.spec)


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
