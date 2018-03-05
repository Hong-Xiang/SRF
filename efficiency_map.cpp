
#include "recon_req.h"



#define MAX_OUTPUTS 100

void RayDrBP_SM(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, float *sinogram, float *volume, int N[3], int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp);


int main()
{ 


	int n;
	FILE *fid, *fid_eff;
	char eff_fname[128], sysmat_fname[128];
	float *sinogram, *eff_map;
	int seg_ind, ns, nchunk;
	int *Nchunk;
	int *Nchunk_acc;
	int *Itemp;
	short *SMtemp;
	int initial_numz,*Znum, *Znuma;
	int Zlength=0;

	int chunk;
	long non_o_sm=0;
	int ind;
	System_info PET;
	VolumeParameter Volpara;
	ReconParameter Reconpara;

	
	

/************************************************************************************************/
	
	strcpy(sysmat_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_11_28_2017\\sm_new3");
	strcpy(eff_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_11_28_2017\\data_new3.eff");
	/*
	if((fid = fopen(sysmat_fname, "rb"))==NULL)
	{ 
		fprintf(stderr, "Could not open sysmat file \"%s\" for reading\n", sysmat_fname);
		exit(1); 
	}
	
	fseek(fid, -1L*sizeof(int), SEEK_END);
	fread(&non_o_sm,size_int,1,fid);
	fclose(fid);
	printf("%d\n",non_o_sm);
	*/
	
	non_o_sm=12007534;
/***********************************************************************************************/
	if((fid = fopen(sysmat_fname, "rb"))==NULL)
	{ fprintf(stderr, "Could not open sysmat file \"%s\" for reading\n", sysmat_fname);
	exit(1); }
		
	fread(&PET, sizeof(System_info), 1, fid);
	fread(&Volpara, sizeof(VolumeParameter), 1, fid);
	fread(&Reconpara, sizeof(ReconParameter), 1, fid);

	Nchunk=ivector(Reconpara.Nsinogram*Reconpara.Nseg);
	Nchunk_acc=ivector(Reconpara.Nsinogram*Reconpara.Nseg);
	Itemp=ivector(non_o_sm);
	SMtemp=sivector(non_o_sm);



	for(seg_ind=0;seg_ind<Reconpara.Nseg;seg_ind++)
	{
		printf("seg_ind=%d\n",seg_ind);

		for(ns=0;ns<Reconpara.Nsinogram;ns++)
		{
			// read volume index and compare to expected
			fread(&n, sizeof(int), 1, fid);
			if(n!=(ns+seg_ind*Reconpara.Nsinogram))
			{
				fprintf(stderr, "Read in sinogram index %d not equal to expected %d\n", n, ns);
				exit(1);
			}
			// fread chunk size and indices and SM chunks
			fread(&Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns], size_int, 1, fid);
			fread(&Nchunk[seg_ind*Reconpara.Nsinogram+ns], size_int, 1, fid);


			fread(Itemp+Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns], size_int, Nchunk[seg_ind*Reconpara.Nsinogram+ns], fid);
			fread(SMtemp+Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns], sizeof(short),Nchunk[seg_ind*Reconpara.Nsinogram+ns], fid);
		}
	}
	fclose(fid);
/*******************************************************************************************************************************/

	Znum=ivector(Reconpara.Nseg);
	Znuma=ivector(Reconpara.Nseg);
	for (seg_ind=0;seg_ind<Reconpara.Nseg;seg_ind++)
	{

		//initial_numz=max(0, (seg_ind+1)/2*Reconpara.SPAN-(Reconpara.SPAN-1)/2);
		initial_numz=max(0, (seg_ind+1)/2);

		Znum[seg_ind]=PET.Ring_num-initial_numz;

		Znuma[seg_ind]=Zlength;
		Zlength +=Znum[seg_ind];
		printf("Znum=%d\n",Znum[seg_ind]);
	}


/**************************************************************************************************************************************/	
  // allocations
	sinogram = (float *) calloc(size_float, Reconpara.Nsinogram*Reconpara.SYM*Zlength);
	eff_map  = (float *) calloc(size_float, Reconpara.Nvolume); 
	
	f1Dassign(sinogram, Reconpara.Nsinogram*Reconpara.SYM*Zlength, 1);
	RayDrBP_SM(Reconpara, Zlength, Znum, Znuma, sinogram, eff_map, Volpara.num, Nchunk, Nchunk_acc, Itemp, SMtemp); 

	for(n=0; n<Reconpara.Nvolume; n++)
	{
	 // printf("%f\n",eff_map[n]);
		if(eff_map[n] > realmin)  eff_map[n] = 1./ eff_map[n];
		else eff_map[n] = realmax;

	//	printf("%f\n",eff_map[n]);
	}


		
	if((fid = fopen(eff_fname, "wb"))==NULL)  
	{ 
		fprintf(stderr, "Could not open efficiency map file \"%s\" for writing\n", eff_fname);
		exit(1);
	}
	
	fwrite(eff_map, sizeof(float), Reconpara.Nvolume, fid);
	fclose(fid);

/*****************************************************************************************************************************************/		
	
	free(Nchunk);
	free(Nchunk_acc);
	free(Itemp);
	free(SMtemp);
	free(Znum);
	free(Znuma);
	free(sinogram);
	free(eff_map);

	return 1;

}







void RayDrBP_SM(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, float *sinogram, float *volume, int N[3], int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp)
{
  int ns, n;
  int z_ind;
  int Itemp_new, ns_new;
  int sym_ind;
  int seg_ind;
  
  memset(volume, 0, size_float*Reconpara.Nvolume);

  for (seg_ind=0;seg_ind<Reconpara.Nseg;seg_ind++)
  {
 	  for (z_ind=0;z_ind<Znum[seg_ind];z_ind++)
	  {
		  for (sym_ind=0;sym_ind<Reconpara.SYM;sym_ind++)
		  {
			  for(ns=0;ns<Reconpara.Nsinogram;ns++)
			  {
				  ns_new=Comsinlocation(ns,Reconpara.Nsinogram, Reconpara.SYM, sym_ind, z_ind, Znuma[seg_ind]);
				  
				  if (ns_new<(Reconpara.Nsinogram*Reconpara.SYM*Zlength))
				  {
		

				  if (sinogram[ns_new]>realmin)
				  {
					  for (n=0;n<Nchunk[seg_ind*Reconpara.Nsinogram+ns];n++)
				      {
						  Itemp_new=Comimglocation(Itemp[Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns]+n], N,z_ind,sym_ind, Reconpara.SYM_Z);
						  if (Itemp_new<Reconpara.Nvolume)
						  volume[Itemp_new] +=sinogram[ns_new]*(double(SMtemp[Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns]+n])/255.0);
						 
					  }
				  }
				  }
			  }
		  }
	  }
  }

  return;
}




