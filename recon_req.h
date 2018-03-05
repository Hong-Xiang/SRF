#include "pet_req.h"

#define SPD 3E10  //light speed (cm/s)

struct ReconParameter
{
	float R_FOV;
	float Z_FOV;
	int SYM;  // symmerty of the system matrix
	int SYM_Z; // symmerty of system matrix in axis direction
	int Nseg;  // number of segment
	int Nsinogram; // pixels number of sinogram 
	int Nvolume; // voxels number of reconstructed image
	TOF_info TOFinfo;// TOF information
}; ReconParameter;

struct VolumeParameter
{
	int num[3]; // voxels number in X Y Z axis
	float detla[3]; // voxels size in X Y Z axis (mm)
	float org[3]; // coordinate origin of the volume in X Y Zaxis (mm)

}; VolumeParameter;

struct SinogramParameter
{
	int dis;
	int angle;
	int slice;
}; SinogramParameter;

int Comimglocation(int oldloc, int N[3], int z_ind, int sym_ind, int SYM_Z);
int Comsinlocation(int oldloc, int Nsinogram, int SYM, int sym_ind, int z_ind, int Znuma);

void CalculateFactor_3D(double *sou_p, double *end_p, VolumeParameter VolPara, float ***volume);
void CalculateFactor_2D(double *sou_p, double *end_p, VolumeParameter VolPara, float **volume);
void CalculateFactor_siddon(double *in_p, double *end_p, VolumeParameter VolPara, float ***volume);
void Location2coord(System_info PET, int location, int ring_ind, float delta_z, double *coord);
void CalculateFactor_TOF(double *sou_p, double *end_p, float TOF_dif_time, TOF_info TOFinfo, VolumeParameter VolPara, float ***volume);
void list2sino(System_info PET, ReconParameter Reconpara, SinogramParameter Sinopara, char listfile_name[128], short *sinogram);