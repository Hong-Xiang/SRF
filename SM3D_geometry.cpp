
#include "recon_req.h"


void main()
{
	
	VolumeParameter Volpara;
	ReconParameter Reconpara;
	System_info PET;

	int i;
	char *optarg;
	int *indices;
	float *entries;

	int seg_ind, phi_ind, dis_ind;
	double detla_dis, detla_phi, detla_z;
	//double in_p_mid[3], end_p_mid[3];
	double *in_p, *end_p;
	int loc_1, loc_2, mid_loc_1, mid_loc_2;

	int DIS, PHI;

	int sm_index=0;
	FILE *fid;
	char sm_name[128];
	float ***volume;
	int total;
	int total_all=0;
	int ind;

	int sli_ind;
	int initial_numz;

	int mid_seg_id;


	int ind_0,ind_1,ind_2;
	float volume_sum;
	
   // for (i = 0; i < argc; i++) printf("%s%s", argv[i], (i < argc-1) ? " " : "\n");
    
  //  while ((i = getopt(argc, argv, "a:e:i:I:N:o:p:P:r:s:v:w:h")) != -1) {
  //      switch (i) {
  //          case 'X':/* max number of crystal per ring */
  //              sscanf(optarg, "%d", &Xtal);
  //              break;
  //          case 'R': /* radius of ring */
  //              sscanf(optarg, "%f", &R);
  //              break;
  //			case 'F': /* radius of FOV */
  //              sscanf(optarg, "%f", &R_FOV);
  //              break;
  //			case 'D': /* Distance range */
  //              sscanf(optarg, "%d", &DIS);
  //              break;
  //          case 'S': /* out filename */
  //              strcpy(sm_name, optarg);
  //             break;
  //         default:
  //              fprintf(stderr,"Could not recognize the input parameter!");
  //      }
  //  }
	

/*******************************************************************/
/*system parameters
	PET.R = 11.85;// 9.7;//9.7;//9.7;//41.2;//8.05; radial radius of the system
	PET.Z_length = 0.159;// 3.36;//13.44;//3.36;//13.44;//6.72;//10.08;//13.44;//3.36;//15.52;//2.91;//12.7;//12.7; axial length of the system
	PET.Ring_num = 80;// 10;//80;//10;//30;//80;//10;//32;//32;//80;//80;   Number of rings
	PET.Blk_num=19;  // Number of blocks in each ring
	PET.Xtal_blk = 20;// 10;//10;//20; // number of xtals in each blocks (radial direction)
	PET.Xtal_num=PET.Blk_num*PET.Xtal_blk;//320;//160; Number of xtals in each ring
	PET.Xtal_size = 0.15;// 0.3;//0.15;//0.168;//0.318; //unit: cm
	PET.Xtal_gap = 0.008;// 0.036;//0.043;  //unit: cm
	PET.Xtal_length=13.44;
	PET.DOI_info.Bin_Num=1;  //
	PET.DOI_info.DOI_reso=5; //
/********************************************************************/
/*Inveon system parameters*/
	PET.R = 8.05;// 9.7;//9.7;//9.7;//41.2;//8.05; radial radius of the system
	PET.Z_length = 12.7;// 3.36;//13.44;//3.36;//13.44;//6.72;//10.08;//13.44;//3.36;//15.52;//2.91;//12.7;//12.7; axial length of the system
	PET.Ring_num = 1;// 10;//80;//10;//30;//80;//10;//32;//32;//80;//80;   Number of rings
	PET.Blk_num = 16/2;  // Number of blocks in each ring
	PET.Xtal_blk = 20/4;// 10;//10;//20; // number of xtals in each blocks (radial direction)
	PET.Xtal_num = PET.Blk_num*PET.Xtal_blk;//320;//160; Number of xtals in each ring
	PET.Xtal_size = 0.159*4;// 0.3;//0.15;//0.168;//0.318; //unit: cm
	PET.Xtal_gap = 0.000;// 0.036;//0.043;  //unit: cm
	PET.Xtal_length = 10;
	PET.DOI_info.Bin_Num = 1;  //
	PET.DOI_info.DOI_reso = 10; //
/*******************************************************************/
/*system parameters for our Micro-PET
	PET.R=5.23;//9.7;//41.2;//8.05; radial radius of the system
	PET.Z_length=14.2;//3.36;//15.52;//2.91;//12.7;//12.7; axial length of the system
	PET.Ring_num=80;//10;//32;//32;//80;//80;   Number of rings
	PET.Blk_num=10;  // Number of blocks in each ring
	PET.Xtal_blk=20; // number of xtals in each blocks (radial direction)
	PET.Xtal_num=PET.Blk_num*PET.Xtal_blk;//320;//160; Number of xtals in each ring
	PET.Xtal_size=0.15; //unit: cm
	PET.Xtal_gap=0.02;  //unit: cm
	PET.DOI_info.Bin_Num=0;  //
	PET.DOI_info.DOI_reso=0; //
/********************************************************************/




	//strcpy(sm_name,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_10_25_2017\\jaszczak1.5-1.5-5rotate21_test160_160slices_sym_4.sm");
	strcpy(sm_name,"E:\\recon_1_10_2018\\sm\\sm_1_24_2018_8x");
/*******************************************************/
	/*sinogram parameters*/
	Reconpara.R_FOV=sqrt(2.)/2*PET.R;//R*0.7071;
	printf("R_fov=%f\n",Reconpara.R_FOV);
	Reconpara.Z_FOV=PET.Z_length;//15.52;//2.91;//10.2133;//15.0806;
	Reconpara.SYM=1;
	Reconpara.SYM_Z=1;
	Reconpara.TOFinfo.Time_reso=0;
	//Reconpara.Nseg=2*((PET.Ring_num-(Reconpara.SPAN+1)/2)/Reconpara.SPAN)+1;
	Reconpara.Nseg=2*PET.Ring_num-1;

	PHI = 320/8;// PET.Xtal_num / 2 / Reconpara.SYM;
	DIS=128/8;

	detla_z=PET.Z_length/PET.Ring_num;
/*******************************************************************/
	/*image parameters*/
	Volpara.num[0] = 128/8;// 80;//160;
	Volpara.num[1] = 128/8;// 80;//160;
	Volpara.num[2] = 1;// 2 * PET.Ring_num;


	Volpara.detla[0] = 0.159*8;// *Reconpara.R_FOV / Volpara.num[0];
	Volpara.detla[1] = 0.159*8;// 4 * Reconpara.R_FOV / Volpara.num[1];
	Volpara.detla[2] = Reconpara.Z_FOV/ PET.Ring_num/Reconpara.SYM_Z;


	Volpara.org[0] = -((Volpara.num[0]) / 2.0)*Volpara.detla[0];
	Volpara.org[1] = -((Volpara.num[1]) / 2.0)*Volpara.detla[1];
	
	Volpara.org[2] = -(float(Volpara.num[2]) / 2.0)*Volpara.detla[2];

/*******************************************************************/
	Reconpara.Nvolume=Volpara.num[0]*Volpara.num[1]*Volpara.num[2];
	Reconpara.Nsinogram=PHI*DIS;
	Reconpara.TOFinfo.Time_reso=0;
/*******************************************************************/
   
	volume= f3Dmatrix(Volpara.num[0], Volpara.num[1], Volpara.num[2]);

/******************************************************************/
	if( (fid=fopen(sm_name, "wb"))==NULL) // write system matrix
  	{
		fprintf(stderr, "Failed to open system matrix file %s for writing.\n", sm_name);
		exit(1);
	}
	fwrite(&PET, sizeof(System_info), 1, fid);
	fwrite(&Volpara, sizeof(VolumeParameter), 1, fid);
	fwrite(&Reconpara, sizeof(ReconParameter), 1, fid);
	
	printf("Generating sm for the all %d segments \n",Reconpara.Nseg);
	for (seg_ind=0;seg_ind<Reconpara.Nseg;seg_ind++)
	{
		printf("Segment No.%d\n",seg_ind);
		initial_numz=max(0, (seg_ind+1)/2);

	//	for (sli_ind=0;sli_ind<(PET.Ring_num-initial_numz);sli_ind++)
	//	{
		//	printf("slid_id=%d\n", sli_ind);
		sli_ind = 0;
		
		for (phi_ind=0;phi_ind<PHI;phi_ind++)
		{
			printf("phi_ind=%d\n",phi_ind);
			

			for (dis_ind=0;dis_ind<DIS;dis_ind++)
			{
/*****************************************************************************************************************************************/

/*******************************************************************************************************************************************/
				//printf("dis_ind=%d\n",dis_ind);
				mid_loc_1=phi_ind;
				mid_loc_2=phi_ind+PET.Xtal_num/2;

				/*
				loc_1=mid_loc_1+floor((dis_ind-DIS/2)/2.0);  
				loc_2=mid_loc_2-floor((dis_ind-DIS/2+1)/2.0);
				*/

				loc_1 = mid_loc_1 + floor(dis_ind - DIS / 2);
				loc_2 = mid_loc_2 - floor(dis_ind - DIS / 2 + 1);



				if (loc_1>=PET.Xtal_num) loc_1-=PET.Xtal_num;
				if (loc_2>=PET.Xtal_num) loc_2-=PET.Xtal_num;
				
				if (loc_1<0) loc_1+=PET.Xtal_num;
				if (loc_2<0) loc_2+=PET.Xtal_num;

				in_p =(double *) calloc(3, sizeof(double));
				end_p =(double *) calloc(3, sizeof(double));

				if (seg_ind%2==0)
				{
					mid_seg_id=(seg_ind+1)/2;
					Location2coord(PET, loc_1, sli_ind+mid_seg_id, detla_z, in_p);
					Location2coord(PET, loc_2, sli_ind, detla_z, end_p);
				}
				else
				{
					mid_seg_id=(seg_ind+1)/2;
					Location2coord(PET, loc_1, sli_ind, detla_z, in_p);
					Location2coord(PET, loc_2, sli_ind+mid_seg_id, detla_z, end_p);
				}



/*******************************************************************************************************************************************/
				f3Dassign(volume, Volpara.num[0], Volpara.num[1], Volpara.num[2], 0);
				
				indices=ivector(Volpara.num[0]*Volpara.num[1]*Volpara.num[2]);
				
				entries =(float *) calloc(Volpara.num[0]*Volpara.num[1]*Volpara.num[2], sizeof(float));

				total=0;
				fwrite(&sm_index , 1 , sizeof(int) , fid );
				fwrite(&total_all , 1 , sizeof(int) , fid );

				CalculateFactor_3D(in_p, end_p, Volpara, volume);
				
				/*
				volume_sum=0;


				for (ind_0=0;ind_0<Volpara.num[0];ind_0++)
				{
					for(ind_1=0;ind_1<Volpara.num[1];ind_1++)
					{
						for (ind_2=0;ind_2<Volpara.num[2];ind_2++)
						{
							volume_sum+=volume[ind_2][ind_1][ind_0];
						}
					}
				}
				volume_sum=1./(volume_sum+realmin);

				for (ind_0=0;ind_0<Volpara.num[0];ind_0++)
				{
					for(ind_1=0;ind_1<Volpara.num[1];ind_1++)
					{
						for (ind_2=0;ind_2<Volpara.num[2];ind_2++)
						{
							volume[ind_2][ind_1][ind_0]*=volume_sum;
						}
					}
				}
				*/
				
				
				total=extract_nonzero_entries3D(volume, Volpara.num[0], Volpara.num[1], Volpara.num[2], indices, entries);
				total_all+=total;
				fwrite(&total , 1 , sizeof(int) , fid );

				
				if(total)
				{
					fwrite(indices , total , sizeof(int) , fid );
					fwrite(entries , total , sizeof(float) , fid );		
				}
				sm_index++;
				
				/*
				for (ind=0;ind<total;ind++)
				{
		
				
 
				printf("indices[%d]=%d\n",ind,indices[ind]);
				printf("entries[%d]=%d\n",ind,entries[ind]);	
					
				}
			*/
				free(indices);
				free(entries);	
				free(in_p);
				free(end_p);
			}			
		}
   // }
	
	}
	



	fwrite(&total_all,1,sizeof(int), fid);
	printf("total_all=%ld\n",total_all);
	fclose(fid);
	free3Dfmatrix(volume,Volpara.num[0],Volpara.num[1],Volpara.num[2]);
}


