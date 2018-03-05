
#include "recon_req.h"


#define N_ITERATIONS 100
// Max iterations and outputs of different iterations
#define MAX_ITERATIONS 1000
#define MAX_OUTPUTS 100
//#define ATAC
//#define NORM

void MLEM_recon(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, short *sinogram, float *volume, int N[3], int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp, float *atan_sino, float *norm_map, int LastIteration, int *IterationsForOutput, char *image_fname);
void RayDrBP_SM(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, float *sinogram, float *volume, int N[3], int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp, float *atan_sino, float *norm_map);
void RayDrFP_SM(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, float *sinogram, float *volume, int N[3], int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp, float *atan_sino, float *norm_map);
void atan_sm(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, float *atan_map, int N[3], int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp, float *atan_sino);
void image_blurmap(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, int N[3], int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp, float *atan_sino, float *norm_map, float **image_blur);


int main()
{ 


	int n;
	int LastIteration;
	int IterationsForOutput[MAX_OUTPUTS];
	FILE *fid, *fid_sino, *fid_atan, *fid_norm;
	char sino_fname[128], image_fname[128], sysmat_fname[128],atan_fname[128], norm_fname[128];
	short *sinogram;
	float *volume;
	float *atan_map;
	float *atan_sino;
	float *norm_map;
	int seg_ind, ns, nchunk;
	int *Nchunk;
	int *Nchunk_acc;
	int *Itemp;
	short *SMtemp;
	int initial_numz,*Znum, *Znuma;
	int Zlength=0;

	int chunk;
	int acc_ind;
	int non_o_sm=0;
	int ind;
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
	
	//strcpy(sysmat_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_10_25_2017\\jaszczak1.5-1.5-5rotate21_test160_160slices_sym_4.sm");
	//strcpy(sino_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_10_25_2017\\jaszczak1.5-1.5-5rotate_160_2_test.s");
	//strcpy(image_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_10_25_2017\\jaszczak1.5-1.5-5rotate_160_2_sym_4_test.re");
	
	strcpy(sysmat_fname,"E:\\recon_1_10_2018\\sm\\sm");
	strcpy(sino_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_11_28_2017\\all_sino_new");
	strcpy(image_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_11_28_2017\\all_sino_new8_norm¡ª¡ªtest.re");




	
	if((fid = fopen(sysmat_fname, "rb"))==NULL)
	{ 
		fprintf(stderr, "Could not open sysmat file \"%s\" for reading\n", sysmat_fname);
		exit(1); 
	}
	
	fseek(fid, -sizeof(int), SEEK_END);
	fread(&non_o_sm,sizeof(int),1,fid);
	fclose(fid);
	
	//non_o_sm=12020272;
	//non_o_sm=20520594;
	//non_o_sm=8989370;
	//non_o_sm=10720279;
	//non_o_sm=11296074;
	printf("non_o_sm=%d\n",non_o_sm);
	//non_o_sm=817047750;
	//non_o_sm=949880191;
/***********************************************************************************************/
	if((fid = fopen(sysmat_fname, "rb"))==NULL)
	{ fprintf(stderr, "Could not open sysmat file \"%s\" for reading\n", sysmat_fname);
	exit(1); }
		
	fread(&PET, sizeof(System_info), 1, fid);
	fread(&Volpara, sizeof(VolumeParameter), 1, fid);
	fread(&Reconpara, sizeof(ReconParameter), 1, fid);
	//Reconpara.SYM_Z=2;
//	Reconpara.Nseg=159;
//	Reconpara.Nsinogram=100*160;
//	Reconpara.Nvolume=80*80*79;
	Nchunk=ivector(Reconpara.Nsinogram*Reconpara.Nseg);
	Nchunk_acc=ivector(Reconpara.Nsinogram*Reconpara.Nseg);
	Itemp=ivector(non_o_sm);
	SMtemp=sivector(non_o_sm);



	for(seg_ind=0;seg_ind<Reconpara.Nseg;seg_ind++)
	{
		//printf("seg_ind=%d\n",seg_ind);

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

			/*
			for (acc_ind=0;acc_ind<(seg_ind*Reconpara.Nsinogram+ns);acc_ind++)
			{
				Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns]+=Nchunk[acc_ind];
			}
			*/

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

		initial_numz=max(0, (seg_ind+1)/2);

		Znum[seg_ind]=PET.Ring_num-initial_numz;
		
		Znuma[seg_ind]=Zlength;
		Zlength +=Znum[seg_ind];
		printf("Znuma=%d\n",Znuma[seg_ind]);
	}
/**************************************************************************************************************************************/
/*attenuation map*/
#ifdef ATAC
	strcpy(atan_fname, "");
	atan_map=fvector(Reconpara.Nvolume);
	atan_sino = fvector(Reconpara.Nsinogram*Reconpara.SYM*Zlength);
	if((fid_atan = fopen(atan_fname, "rb"))==NULL)
	{ 
		fprintf(stderr, "Could not open attenuation map file \"%s\" for reading\n", atan_fname);
		exit(1); 
	}
	fread(atan_map,sizeof(float),Reconpara.Nvolume,fid_atan);
	fclose(fid_atan);
	atan_sm(Reconpara, Zlength, Znum, Znuma, atan_map, Volpara.num, Nchunk, Nchunk_acc,Itemp,SMtemp,atan_sino);
#else
	atan_sino = fvector(Reconpara.Nsinogram*Reconpara.SYM*Zlength);
	f1Dassign(atan_sino, Reconpara.Nsinogram*Reconpara.SYM*Zlength, 0);
#endif
/*****************************************************************************************************************************************/
/*normalization map*/
#ifdef NORM
	strcpy(norm_fname, "E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_11_28_2017\\all_sino_sm_new3.re.iter020");
	norm_map = fvector(Reconpara.Nvolume);
	if ((fid_norm = fopen(norm_fname, "rb")) == NULL)
	{
		fprintf(stderr, "Could not open normalization file \"%s\" for reading\n", norm_fname);
		exit(1);
	}
	fread(norm_map, sizeof(float), Reconpara.Nvolume, fid_norm);
	fclose(fid_norm);
#else
	norm_map = fvector(Reconpara.Nsinogram*Reconpara.SYM*Zlength);
	f1Dassign(norm_map, Reconpara.Nsinogram*Reconpara.SYM*Zlength, 1.0);
#endif

/**************************************************************************************************************************************/

  // allocations
	sinogram = (short *) calloc(sizeof(short), Reconpara.Nsinogram*Reconpara.SYM*Zlength);
	volume   = (float *) calloc(size_float, Reconpara.Nvolume); 


	
	if((fid_sino = fopen(sino_fname, "rb"))==NULL)
	{ 
		fprintf(stderr, "Could not open sinogram data file \"%s\" for reading\n", sino_fname);
		exit(1);
	}

	fread(sinogram,sizeof(short),Reconpara.Nsinogram*Reconpara.SYM*Zlength,fid_sino);
	fclose(fid_sino);

	

	fprintf(stderr, "Forward sinogram of length %d into image of length %d\n", Reconpara.Nsinogram*Reconpara.SYM*Zlength, Reconpara.Nvolume);
		
	MLEM_recon(Reconpara, Zlength, Znum, Znuma, sinogram, volume, Volpara.num, Nchunk, Nchunk_acc, Itemp, SMtemp, atan_sino, norm_map,LastIteration, IterationsForOutput, image_fname); 
	

	free(sinogram);
	free(volume);

	free(Nchunk);
	free(Nchunk_acc);
	free(Itemp);
	free(SMtemp);
	free(Znum);
	free(Znuma);


	return 1;

}




void MLEM_recon(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, short *sinogram, float *volume, int N[3], int *Nchunk, int *Nchunk_acc,int *Itemp, short *SMtemp, float *atan_sino, float *norm_map,int LastIteration, int *IterationsForOutput, char *image_fname)
{
  FILE *fid;
  int n, niter,index,ind;
  float *tmp_volume, *tmp_sinogram, *constant_denominator;
  char temp_fname[128];
  float **image_blur;
 // size_t size_int, size_float;
  image_blur=f2Dmatrix(Reconpara.Nvolume, Reconpara.Nvolume);

  // --- initial volume assignment: all pixels are one
  f1Dassign(volume, Reconpara.Nvolume, 1.0);
// ------ create ml_em variables:
  tmp_sinogram=fvector(Reconpara.Nsinogram*Reconpara.SYM*Zlength);
  constant_denominator=fvector(Reconpara.Nvolume);
  tmp_volume=fvector(Reconpara.Nvolume);

// image_blurmap(Reconpara, Zlength, Znum, Znuma, N[3], Nchunk, Nchunk_acc, Itemp, SMtemp, atan_sino, norm_map, image_blur);
// --- compute  element-by-element inverse of efficiency matrix
  f1Dassign(tmp_sinogram, Reconpara.Nsinogram*Reconpara.SYM*Zlength, 1);

  RayDrBP_SM(Reconpara, Zlength, Znum, Znuma, tmp_sinogram, constant_denominator, N, Nchunk, Nchunk_acc, Itemp, SMtemp,atan_sino, norm_map); 

	 
  for(n=0; n<Reconpara.Nvolume; n++)
  {
  //  printf("%f\n",constant_denominator[n]);
	if(constant_denominator[n] > realmin)  constant_denominator[n] = 1./ constant_denominator[n];
    else constant_denominator[n] = realmax;
  }
  
//  -------- ITERATION LOOP --------
  for(niter=1; niter<=LastIteration; niter++)
  {
    fprintf(stderr, "Iteration No %d of %d\n", niter, LastIteration); 
	// compute the reprojection through the n-1 version of the file into tmp_sinogram
	
	RayDrFP_SM(Reconpara, Zlength, Znum, Znuma, tmp_sinogram, volume, N, Nchunk, Nchunk_acc, Itemp, SMtemp, atan_sino, norm_map); 
	
// divide the sinogram by the tmp_sinogram
	
    for(n=0; n<Reconpara.Nsinogram*Reconpara.SYM*Zlength; n++)
    {
		if(sinogram[n] <= 0.)  
		{
			tmp_sinogram[n] = 0.;
		}  
		else
		{
           if(tmp_sinogram[n] > realmin) 
			   tmp_sinogram[n] = float(sinogram[n])/255.0 / tmp_sinogram[n];
           else                          
			   tmp_sinogram[n] = float(sinogram[n])/255.0 * realmax;
		}
	}
	
	// backproject the result inot tmp_volume
      RayDrBP_SM(Reconpara, Zlength, Znum, Znuma, tmp_sinogram, tmp_volume,N, Nchunk, Nchunk_acc,Itemp, SMtemp, atan_sino,norm_map); 
	  
// multiply by the constant denominator
	 
    for(n=0; n<Reconpara.Nvolume; n++)
    {
		volume[n] *= constant_denominator[n] * tmp_volume[n];// / (norm_map[n] + realmin);

		if(volume[n] < 0.)
		  volume[n] = 0.;
    }
	
	

	if(niter==IterationsForOutput[niter])
    {
		sprintf(temp_fname,"%s.iter%03d",image_fname,niter);
		if((fid = fopen(temp_fname, "wb"))==NULL)  
			{ fprintf(stderr, "Could not open image file \"%s\" for writing\n", temp_fname);
			exit(1);}
		fwrite(volume, sizeof(float), Reconpara.Nvolume, fid);
		fclose(fid);
	}


  }
  // end: free up memory
  free(tmp_sinogram);
  free(tmp_volume);
  free(constant_denominator);
  
  return;
}


void RayDrBP_SM(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, float *sinogram, float *volume, int N[3], int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp, float *atan_sino, float *norm_map)
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
							  Itemp_new=Comimglocation(Itemp[Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns]+n], N,z_ind,sym_ind,Reconpara.SYM_Z);

							  if (Itemp_new<Reconpara.Nvolume)
							  {
								  volume[Itemp_new] +=sinogram[ns_new]*(float(SMtemp[Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns]+n])/255.0);
								  //printf("%f\n",(float(SMtemp[Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns]+n])/255.0));
							  }
						}
					}
				}
			}
		  }
	  }
  }

  return;
}


void RayDrFP_SM(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, float *sinogram, float *volume, int N[3], int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp, float *atan_sino, float *norm_map)
{
  int ns, n;
  int z_ind;
  int Itemp_new, ns_new;
  int sym_ind;
  int seg_ind;

  memset(sinogram, 0, size_float*Reconpara.Nsinogram*Reconpara.SYM*Zlength);
  
  for (seg_ind=0;seg_ind<Reconpara.Nseg;seg_ind++)
  {
	  for (z_ind=0;z_ind<Znum[seg_ind];z_ind++)
	  {
		  for (sym_ind=0;sym_ind<Reconpara.SYM;sym_ind++)
		  {
			  for(ns=0;ns<Reconpara.Nsinogram;ns++)
			  {
				  ns_new=Comsinlocation(ns, Reconpara.Nsinogram, Reconpara.SYM, sym_ind, z_ind, Znuma[seg_ind]);
				  
				  if (ns_new<(Reconpara.Nsinogram*Reconpara.SYM*Zlength))
				  {
	
						for (n=0;n<Nchunk[seg_ind*Reconpara.Nsinogram+ns];n++)
						{
							Itemp_new=Comimglocation(Itemp[Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns]+n], N,z_ind,sym_ind, Reconpara.SYM_Z);
					  
							if (Itemp_new<Reconpara.Nvolume)
							{
								  sinogram[ns_new] += volume[Itemp_new]*(float(SMtemp[Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns]+n])/255.0);
							}					  


		   			   }
				}
			  }
		  }
	  }
  }
  
  return;
}





void atan_sm(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, float *atan_map, int N[3], int *Nchunk, int *Nchunk_acc,int *Itemp, short *SMtemp, float *atan_sino)
{
	
	int ns,n;
	int z_ind;
	int Itemp_new, ns_new;
	int sym_ind;
	int seg_ind;


	for (seg_ind=0;seg_ind<Reconpara.Nseg;seg_ind++)
	{
		for (z_ind=0;z_ind<Znum[seg_ind];z_ind++)
		{
			for (sym_ind=0;sym_ind<Reconpara.SYM;sym_ind++)
			{
				for(ns=0;ns<Reconpara.Nsinogram;ns++)
				{
					ns_new=Comsinlocation(ns, Reconpara.Nsinogram, Reconpara.SYM, sym_ind, z_ind, Znuma[seg_ind]);
				  
					if (ns_new<(Reconpara.Nsinogram*Reconpara.SYM*Zlength))
					{
						for (n=0;n<Nchunk[seg_ind*Reconpara.Nsinogram+ns];n++)
						{
							Itemp_new=Comimglocation(Itemp[Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns]+n], N,z_ind,sym_ind, Reconpara.SYM_Z);
					  
							if (Itemp_new<Reconpara.Nvolume)
							{
								atan_sino[ns_new] += atan_map[Itemp_new]*(float(SMtemp[Nchunk_acc[seg_ind*Reconpara.Nsinogram+ns]+n])/255.0);	
							}					  
		   				}
					}
				}
			}
		}
	}
}


void image_blurmap(ReconParameter Reconpara, int Zlength, int *Znum, int *Znuma, int N[3], int *Nchunk, int *Nchunk_acc, int *Itemp, short *SMtemp,float *atan_sino, float *norm_map, float **image_blur)
{
	int ind;
	float *volume;
	float *tmp_sinogram;

	volume = fvector(Reconpara.Nvolume);
	tmp_sinogram = fvector(Reconpara.Nsinogram*Reconpara.SYM*Zlength);

	for (ind = 0; ind < Reconpara.Nvolume; ind++)
	{
		volume[ind] = 1.0;

		RayDrFP_SM(Reconpara, Zlength, Znum, Znuma, tmp_sinogram, volume, N, Nchunk, Nchunk_acc, Itemp, SMtemp, atan_sino, norm_map);
		RayDrBP_SM(Reconpara, Zlength, Znum, Znuma, tmp_sinogram, image_blur[ind], N, Nchunk, Nchunk_acc, Itemp, SMtemp, atan_sino, norm_map);
	}
	

}