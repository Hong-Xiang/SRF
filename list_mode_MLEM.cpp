
#include "recon_req.h"


#define N_ITERATIONS 50
// Max iterations and outputs of different iterations
#define MAX_ITERATIONS 1000
#define MAX_OUTPUTS 100

void MLEM_recon(int ls_num, int Nvolume, float *volume, int *Nchunk, int *Nchunk_acc,int *Itemp, short *SMtemp, int LastIteration, int *IterationsForOutput, char *eff_fname, char *image_fname);
void RayDrBP_ls(int ls_num, int Nvolume, float *ls_data, float *volume, int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp);
void RayDrFP_ls(int ls_num, int Nvolume, float *ls_data, float *volume, int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp);



int main()
{ 


	int n;
	int LastIteration;
	int IterationsForOutput[MAX_OUTPUTS];
	FILE *fid;
	char eff_fname[128], image_fname[128], sysmat_fname[128];
	float *volume;
	int nchunk;
	int *Nchunk;
	int *Nchunk_acc;
	int *Itemp;
	short *SMtemp;
	int ls_num;
	int ls_ind;
	int ind;
	int non_o_sm=0;
	int a,b;
	short c;


	System_info PET;
	VolumeParameter Volpara;
	ReconParameter Reconpara;

	LastIteration=N_ITERATIONS;
	//IterationsForOutput[0]=LastIteration;
	for (ind=1;ind<=LastIteration;ind++)
	{
		IterationsForOutput[ind]=ind;
	}
	
	

/************************************************************************************************/
	//strcpy(sysmat_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_8_9_2017\\TOF\\sm_sym_4_160_4_tmp");
	//strcpy(sino_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_8_9_2017\\TOF\\jaszczak3-3-20.ss");
	//strcpy(image_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_8_9_2017\\TOF\\jaszczak3-3-20_4.ssre1");
	
	strcpy(sysmat_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_11_28_2017\\data_200ps.sm");
	strcpy(eff_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_11_28_2017\\data_new3.eff");
	strcpy(image_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_11_28_2017\\data_new.ls_200ps.re");

	if((fid = fopen(sysmat_fname, "rb"))==NULL)
	{ 
		fprintf(stderr, "Could not open sysmat file \"%s\" for reading\n", sysmat_fname);
		exit(1); 
	}
	
	fseek(fid, -sizeof(int), SEEK_END);
	fread(&non_o_sm,size_int,1,fid);
	fclose(fid);
	
	//non_o_sm=1576779006;
/***********************************************************************************************/
	/*
	if((fid = fopen(sysmat_fname, "rb"))==NULL)
	{ 
		fprintf(stderr, "Could not open sysmat file \"%s\" for reading\n", sysmat_fname);
		exit(1); 
	}
		
	fread(&PET, sizeof(System_info), 1, fid);
	fread(&Volpara, sizeof(VolumeParameter), 1, fid);
	fread(&Reconpara, sizeof(ReconParameter), 1, fid);
	//Reconpara.Nvolume=80*80*159;

	fread(&ls_num, size_int, 1, fid);

	Nchunk=ivector(ls_num);
	Nchunk_acc=ivector(ls_num);


	non_o_sm=0;
	a=0;
	c=0;

	for(ls_ind=0;ls_ind<ls_num;ls_ind++)
	{
		//	printf("seg_ind=%d\n",ls_ind);	
			// read volume index and compare to expected
			fread(&n, sizeof(int), 1, fid);
			if(n!=ls_ind)
			{
				fprintf(stderr, "Read in system matrix index %d not equal to expected %d\n", n, ls_num);
				exit(1);
			}
			// fread chunk size and indices and SM chunks
			fread(&Nchunk_acc[ls_ind], size_int, 1, fid);
			fread(&Nchunk[ls_ind], size_int, 1, fid);
			non_o_sm+=Nchunk[ls_ind];

			for (ind=0;ind<	Nchunk[ls_ind];ind++	)
			{
				fread(&a, size_int, 1, fid);
			fread(&c, sizeof(short),1, fid);
			}
	}
	fclose(fid);
	*/



	if((fid = fopen(sysmat_fname, "rb"))==NULL)
	{ 
		fprintf(stderr, "Could not open sysmat file \"%s\" for reading\n", sysmat_fname);
		exit(1); 
	}
		
	fread(&PET, sizeof(System_info), 1, fid);
	fread(&Volpara, sizeof(VolumeParameter), 1, fid);
	fread(&Reconpara, sizeof(ReconParameter), 1, fid);
	//Reconpara.Nvolume=80*80*159;

	fread(&ls_num, size_int, 1, fid);



	Nchunk = ivector(ls_num);
	Nchunk_acc = ivector(ls_num);
	Itemp=ivector(non_o_sm);
	SMtemp=sivector(non_o_sm);


	for(ls_ind=0;ls_ind<ls_num;ls_ind++)
	{
		//	printf("seg_ind=%d\n",ls_ind);	
			// read volume index and compare to expected
			fread(&n, sizeof(int), 1, fid);
			if(n!=ls_ind)
			{
				fprintf(stderr, "Read in system matrix index %d not equal to expected %d\n", n, ls_num);
				exit(1);
			}
			// fread chunk size and indices and SM chunks
			fread(&Nchunk_acc[ls_ind], size_int, 1, fid);
			fread(&Nchunk[ls_ind], size_int, 1, fid);



		fread(Itemp+Nchunk_acc[ls_ind], size_int, Nchunk[ls_ind], fid);
		fread(SMtemp+Nchunk_acc[ls_ind], sizeof(short),Nchunk[ls_ind], fid);
	}
	fclose(fid);
	//printf("b=%d\n",b);
/*******************************************************************************************************************************/



/**************************************************************************************************************************************/	
  // allocations
	volume   = (float *) calloc(size_float, Reconpara.Nvolume); 

	fprintf(stderr, "Forward sinogram of length %d into image of length %d\n", ls_num, Reconpara.Nvolume);
		
	MLEM_recon(ls_num, Reconpara.Nvolume, volume, Nchunk, Nchunk_acc, Itemp, SMtemp, LastIteration, IterationsForOutput, eff_fname, image_fname); 
	
	free(Nchunk);
	free(Nchunk_acc);
	free(Itemp);
	free(SMtemp);

	free(volume);

	return 1;

}




void MLEM_recon(int ls_num, int Nvolume, float *volume, int *Nchunk, int *Nchunk_acc,int *Itemp, short *SMtemp, int LastIteration, int *IterationsForOutput, char *eff_fname, char *image_fname)
{
  FILE *fid, *fid_eff;
  int n, niter,index;
  float *tmp_volume, *tmp_ls_data, *constant_denominator;
  char temp_fname[128];
 // size_t size_int, size_float;

  // --- initial volume assignment: all pixels are one
  f1Dassign(volume, Nvolume, 1);
// ------ create ml_em variables:
  tmp_ls_data=fvector(ls_num);
  constant_denominator=fvector(Nvolume);
  tmp_volume=fvector(Nvolume);


 /****************************************************************************************************************************************/
  /*load efficiency map*/
  if((fid_eff = fopen(eff_fname, "rb"))==NULL)
	{ 
		fprintf(stderr, "Could not open efficiency map file \"%s\" for reading\n", eff_fname);
		exit(1);
	}

	fread(constant_denominator,sizeof(float),Nvolume,fid_eff);
	fclose(fid_eff);


/*****************************************************************************************************************************************/

//  -------- ITERATION LOOP --------
  for(niter=1; niter<=LastIteration; niter++)
  {
    fprintf(stderr, "Iteration No %d of %d\n", niter, LastIteration); 
	// compute the reprojection through the n-1 version of the file into tmp_sinogram
	
	RayDrFP_ls(ls_num, Nvolume, tmp_ls_data, volume, Nchunk, Nchunk_acc, Itemp, SMtemp);
	
// divide the sinogram by the tmp_sinogram

    for(n=0; n<ls_num; n++)
    {
		if(tmp_ls_data[n]>0)
		{
		tmp_ls_data[n]=1.0/tmp_ls_data[n];
		}
		else
			tmp_ls_data[n]=realmax;
	}

	// backproject the result inot tmp_volume
    RayDrBP_ls(ls_num, Nvolume, tmp_ls_data, tmp_volume, Nchunk, Nchunk_acc, Itemp, SMtemp);
	  
// multiply by the constant denominator
    for(n=0; n<Nvolume; n++)
    {
		
		volume[n] *=  constant_denominator[n] * tmp_volume[n];

		if(volume[n] < 0.)
		  volume[n] = 0.;
    }
	
    
	if(niter==IterationsForOutput[niter])
    {
		sprintf(temp_fname,"%s.iter%03d",image_fname,niter);
		if((fid = fopen(temp_fname, "wb"))==NULL)  
			{ fprintf(stderr, "Could not open image file \"%s\" for writing\n", temp_fname);
			exit(1);}
		fwrite(volume, sizeof(float), Nvolume, fid);
		fclose(fid);
	}

  }
  // end: free up memory
  free(tmp_ls_data);
  free(tmp_volume);
  free(constant_denominator);

  
  return;
}


void RayDrBP_ls(int ls_num, int Nvolume, float *ls_data, float *volume, int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp)
{
  int n;
  int ls_ind;
  
  memset(volume, 0, size_float*Nvolume);

 
  for(ls_ind=0;ls_ind<ls_num;ls_ind++)
  {
	  for (n=0;n<Nchunk[ls_ind];n++)
	  {
		 // printf("SMtemp=%f\n",SMtemp[Nchunk_acc[ls_ind]+n]/255.0);
		  volume[Itemp[Nchunk_acc[ls_ind]+n]] +=ls_data[ls_ind]*(double(SMtemp[Nchunk_acc[ls_ind]+n])/255.0);	
		  //printf("Itemp[n]=%d\n",Itemp[n]);
		  //printf("volume=%f\n",volume[Itemp[n]]);
	  }

  }
				  
  return;
}


void RayDrFP_ls(int ls_num, int Nvolume, float *ls_data, float *volume, int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp)
{
  
  int ls_ind;
  int n;

  memset(ls_data, 0, size_float*ls_num);
  
  for (ls_ind=0;ls_ind<ls_num;ls_ind++)
  {
	  for(n=0;n<Nchunk[ls_ind];n++)
	  {
			ls_data[ls_ind] += volume[Itemp[Nchunk_acc[ls_ind]+n]]*(double(SMtemp[Nchunk_acc[ls_ind]+n])/255.0);
	  }
  }		
  
  return;
}





