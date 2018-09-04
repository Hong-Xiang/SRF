from doufo import dataclass


@dataclass
class CastorGeometry():
    nb_rsectors: int
    nb_axial_rsectors: int
    nb_axial_modules: int
    nb_transaxial_modules: int
    nb_axial_submodules: int
    nb_transaxial_submodules: int
    nb_axial_crystals: int
    nb_transaxial_crystals: int

    @property
    def nb_crystals_per_ring(self):
        return (
            self.nb_rsectors*self.nb_transaxial_modules *
            self.nb_transaxial_submodules*self.nb_transaxial_crystals)


def compute_crystal_id(castor_geo: CastorGeometry,
                       id_ring,
                       id_transaxial_crystal,
                       id_transaxial_submodule,
                       id_transaxial_module,
                       id_rsector):
    return (id_ring*castor_geo.nb_crystals_per_ring +
            id_transaxial_crystal +
            id_transaxial_submodule*castor_geo.nb_transaxial_crystals+
            id_transaxial_module*castor_geo.nb_transaxial_submodules*castor_geo.nb_axial_crystals+
            id_rsector*castor_geo.nb_transaxial_modules*castor_geo.nb_transaxial_submodules*castor_geo.nb_transaxial_crystals)


def compute_ring_id(castor_geo,
                    id_axial_crystal,
                    id_axial_submodule,
                    id_axial_module,
                    id_axial_rsector):
    return (id_axial_crystal +
            id_axial_submodule*castor_geo.nb_axial_crystals +
            id_axial_module*castor_geo.nb_axial_submodules*castor_geo.nb_axial_crystals +
            id_axial_rsector*castor_geo.nb_axial_modules*castor_geo.nb_axial_submodules*castor_geo.nb_axial_crystals)
