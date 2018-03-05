#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
//#include <unistd.h>


#define realmin 1.0e-13 //DBL_MIN
#define realmax 1.0e13 //DBL_MAX
#define PI acos(-1.0)
#define NR_END 1
#define size_float sizeof(float)
#define size_int sizeof(int)

struct TOF_info
{
	//int Bin_Num;        // the number of the TOF bin
	//int Min_bin;		//
	//double Detla;		// Detla time of each TOF bin (unit:s)
	double Time_reso;		// Time resolution of the TOF PET system (uint:s)
}; TOF_info;

struct DOI
{
	float DOI_reso;     // DOI resolution: the layer thickness (unit:cm)
	int Bin_Num;        // number of doi layers
}; DOI;

struct System_info
{
	int Ring_num;       // the ring number PET system 
	int Xtal_num;       // the crystsl number in each ring og PET system (ring system)
	int Blk_num;        // the block number per ring (only in Blk System)
	int Xtal_blk;       // the xtals number per block (only in Blk System)
	float Xtal_gap;     // the gap between two xtals in one block (unit:cm) (only in Blk system)
	float Xtal_size;   // the distance between the two xtals in one block (unit:cm)(only in Blk System)
	float Xtal_length; // the length of the xtals
	float R;			// the distance from the origins to the detector blk (unit:cm)	
	float Z_length;     // the length of the PET system in axial. (unit: cm) 
	DOI DOI_info;
}; System_info;

struct Image_package
{
	int Nvolume[3];
	float VOL_size[3];
	float *image;

}; Image_package;

double max(double a, double b);
double min(double a, double b);
void BublleSort (double *arr, int count);//bubble sort algorithm;
int extract_nonzero_entries3D_short(float ***volume, int Nx, int Ny, int Nz, int *indices, short *entries);
int extract_nonzero_entries3D(float ***volume, int Nx, int Ny, int Nz, int *indices, float *entries);
int extract_nonzero_entries2D(float **volume, int Nx, int Ny, int *indices, float *entries);
void nrerror(char error_text[]);
float *fvector(long nxl);
int *ivector(long nxl);
short *sivector(long nxl);
float **f2Dmatrix(long nxl,long nyl);
int **i2Dmatrix(long nxl,long nyl);
short **si2Dmatrix(long nxl,long nyl);
float ***f3Dmatrix(long nxl,long nyl, long nzl);
int ***i3Dmatrix(long nxl,long nyl, long nzl);
float ****f4Dmatrix(long nxl,long nyl, long nzl, long time);
void free2Dfmatrix(float **m, long nxl,long nyl);
void free3Dfmatrix(float ***m, long nxl,long nyl, long nzl);
void free4Dfmatrix(float ****m, long nxl,long nyl, long nzl, long time);
void f1Dassign(float *m, long nxl, float value);
void f2Dassign(float **m, long nxl,long nyl, float value);
void f3Dassign(float ***m, long nxl,long nyl, long nzl, float value);
void i3Dassign(int ***m, long nxl,long nyl, long nzl, int value);
void f2Dwrite(float **m,long nxl, long nyl, FILE *fid);
void f3Dwrite(float ***m,long nxl, long nyl, long nzl, FILE *fid);
void i3Dwrite(int ***m,long nxl, long nyl, long nzl, FILE *fid);
void free3Dimatrix(int ***m, long nxl,long nyl, long nzl);