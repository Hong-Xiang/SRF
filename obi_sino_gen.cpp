
#include "recon_req.h"





void main (int argc, char **argv);

void main (int argc, char **argv) 
{
	char listfile_name[256];  // the input list-mode data name  
	char oblique_sinogram_fname[128]; // the output obilique sinogram data name 
	FILE *fid_w; // file pointer for reading and writing
	int Coin_Num; // the number of the concidences in the listmode data
	int index_coin; // index of the concidence 
	int ind, seg_ind;
	int Ringnum_1; // the ring index for the first photon in each concidence
	int	Ringnum_2; // the ring index for the second photon in each cocidence
	int Ringnum_tmp;
	int Location_1; // the location index per ring for the first photon in each concidence
	int Location_2; // the location index per ring for the second photon in each concidence
	int Location_tmp;

	short *True;
	int Ringnum_diff; // the ring index difference between the two photon in one concidence 
	int Location_diff; //the location index difference between the two photon in one concidence
	int Segment_Num; // the segment index of the concidence
	int OS_Num; // the slice index of the concidence
	int mid_se; //
	System_info PET;
	ReconParameter Reconpara;
	SinogramParameter Sinopara;
/*************************************************************************************************************/	

	float diff_time;
	int Phi_Num;
	int Dis_Num;
	//int out_initial;
	int diff1,diff2,sigma;
	int Znum=0;
	int initial_numz=0;
	int *Znuma;
	
	
	/***********************************input parameters**********************************************************/
	strcpy_s(listfile_name, "E:\\recon_1_18_2018\\jaszczak1.5-1.5-5quarter.ls");
	strcpy_s(oblique_sinogram_fname, "E:\\recon_1_18_2018\\jaszczak1.5-1.5-5quarter.s");

	PET.Ring_num=80; //ND=10;
	PET.Xtal_num=380; 	//MAX_CRYSTAL_NUM=160;
	
	Reconpara.Nseg=2*PET.Ring_num-1;

	Sinopara.angle=PET.Xtal_num/2; 	//MAX_ANGLE=MAX_CRYSTAL_NUM/2; 
	Sinopara.dis=190;      	//MAX_DIS=80;//MAX_CRYSTAL_NUM;
	Sinopara.slice=PET.Ring_num*PET.Ring_num; //2*PET.Ring_num-1; 	//SLICE=ND+ND-1;

	True= (short *) calloc(Sinopara.slice*Sinopara.angle*Sinopara.dis, sizeof(short));

    list2sino(PET, Reconpara, Sinopara,listfile_name, True);
/***********************************************************************************************************************/
/******************************************write to the obilique sinogram***********************************************/
	printf(" Writing sinograms in type of 16-bit Signed Int with size (Distance * Angle * Slice): %d*%d*%d \n",Sinopara.dis, Sinopara.angle, Sinopara.slice);

	fid_w=fopen(oblique_sinogram_fname,"wb");
	fwrite(True,sizeof(short),Sinopara.slice*Sinopara.angle*Sinopara.dis, fid_w);
	free(True);
	fclose(fid_w);		
	
	printf(" Completed! %c%c%c \n",char(64),char(95),char(64));
/****************************************************************************************************************/



}







	



