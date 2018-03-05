
#include "recon_req.h"




void main()
{
	
	VolumeParameter Volpara;
	ReconParameter Reconpara;
	System_info PET;

	int i,n;
	char *optarg;

	int *indices;
	short *entries;

	int seg_ind, phi_ind, dis_ind;
	double detla_dis, detla_phi, detla_z;
	double in_p_mid[3], end_p_mid[3];
	double *in_p, *end_p;

	int DIS, PHI;
	double R_FOV, R, Z_len, Z_FOV;
	int Xtal;

	int sm_index=0;
	FILE *fid, *fid_ls;
	char sm_name[128], ls_fname[128];
	int seg_num, sino_num, volume_num;
	float ***volume;
	int total;
	int total_all=0;
	int ind;
	int TOFind;
	int loc_1, loc_2, ring_1, ring_2;
	float dif_time;
	int ls_ind;
	int ls_num;

	
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
/*system parameters*/
	PET.R=9.7;//9.7;//41.2;//8.05; radial radius of the system
	PET.Z_length=3.36;//13.44;//3.36;//15.52;//2.91;//12.7;//12.7; axial length of the system
	PET.Ring_num=10;//10;//32;//32;//80;//80;   Number of rings
	PET.Blk_num=16;
	PET.Xtal_blk=10;
	PET.Xtal_num=PET.Blk_num*PET.Xtal_blk; // Number of xtals in each ring
	PET.Xtal_size=0.3; //unit: cm
	PET.Xtal_gap=0.036;// unit: cm
	PET.Xtal_length=0.2;// unit: cm
	PET.DOI_info.Bin_Num=1;
	PET.DOI_info.DOI_reso=20; 

	Reconpara.TOFinfo.Time_reso=200E-12;

	DIS=80;
	Volpara.num[0]=80;
	Volpara.num[1]=80;
	
	Reconpara.R_FOV=6.4;//sqrt(2.)/2*PET.R;//R*0.7071;
	Reconpara.Z_FOV=PET.Z_length;//15.52;//2.91;//10.2133;//15.0806;
	Reconpara.SYM=1;

	Reconpara.SYM_Z=2;
	Reconpara.Nseg=0;
	Reconpara.Nsinogram=0;
	//Reconpara.Nseg=2*((PET.Ring_num-(Reconpara.SPAN+1)/2)/Reconpara.SPAN)+1;
	//PHI=PET.Xtal_num/2/Reconpara.SYM;

	

	//PHI=PET.Xtal_num/2/Reconpara.SYM;

	detla_dis=2*Reconpara.R_FOV/DIS; 
	detla_phi=2*PI/PET.Xtal_num;
	detla_z=PET.Z_length/PET.Ring_num;
/*******************************************************************************/


	//strcpy(sm_name,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\ECAT_300mm\\sm_siddon\\sm_ecat_wo_sym_128_128_64_short");
	strcpy(sm_name,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_11_28_2017\\data_200ps.sm");
/*******************************************************************/
/*image parameters*/
	Volpara.num[2]=2*PET.Ring_num-1;//64;//2*Ring;//159;


	Volpara.detla[0] = 2*Reconpara.R_FOV/Volpara.num[0];
	Volpara.detla[1] = 2*Reconpara.R_FOV/Volpara.num[1];
	Volpara.detla[2] = Reconpara.Z_FOV/PET.Ring_num/2;
	
	Volpara.org[0] = -(float(Volpara.num[0]) / 2)*Volpara.detla[0];
	Volpara.org[1] = -(float(Volpara.num[1]) / 2)*Volpara.detla[1];
	Volpara.org[2] = -(float(Volpara.num[2]) / 2)*Volpara.detla[2];
/*******************************************************************/
/*******************************************************************/
	Reconpara.Nvolume=Volpara.num[0]*Volpara.num[1]*Volpara.num[2];
	//Reconpara.Nsinogram=PHI*DIS;
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
	

	strcpy(ls_fname,"E:\\F\\SJTU_project_paper\\Weiji_Tao\\recon_11_28_2017\\data.ls");

	if( (fid_ls=fopen(ls_fname, "rb"))==NULL) // read list-mode data
  	{
		fprintf(stderr, "Failed to open list-mode file %s for reading.\n", ls_fname);
		exit(1);
	}

	fread(&ls_num, size_int, 1, fid_ls);
	

	fwrite(&ls_num, sizeof(int), 1, fid);

	for (ls_ind=0;ls_ind<ls_num;ls_ind++)
	{
		if (ls_ind%(ls_num/100)==0)
		printf(" %d% completed \n",ls_ind/(ls_num/100));

		fread(&n, sizeof(int), 1, fid_ls);
		//printf("n=%d\n",n);
				//printf("read_n=%d\n",tb+ns*Reconpara.TOFinfo.Bin_Num+seg_ind*Reconpara.Nsinogram*Reconpara.TOFinfo.Bin_Num);
		if(n!=ls_ind)
		{
			fprintf(stderr, "Read in list-mode data index %d not equal to expected %d\n", n, ls_ind);
			exit(1);
		}
		
		fread(&ring_1, size_int, 1, fid_ls);
		fread(&ring_2, size_int, 1, fid_ls);
		fread(&loc_1, size_int, 1, fid_ls);
		fread(&loc_2, size_int, 1, fid_ls);
		fread(&dif_time, size_float,1,fid_ls);
		

		in_p =(double *) calloc(3, sizeof(double));
		end_p =(double *) calloc(3, sizeof(double));

		Location2coord(PET, loc_1, ring_1, detla_z, in_p);
		Location2coord(PET, loc_2, ring_2, detla_z, end_p);

		f3Dassign(volume, Volpara.num[0], Volpara.num[1], Volpara.num[2], 0);
				
		indices=ivector(Volpara.num[0]*Volpara.num[1]*Volpara.num[2]);
		entries =(short *) calloc(sizeof(short), Volpara.num[0]*Volpara.num[1]*Volpara.num[2]);
		
		total=0;
		
		fwrite(&sm_index , 1 , sizeof(int) , fid );
					//printf("total_all=%d\n",total_all);
		fwrite(&total_all , 1 , sizeof(int) , fid );

		CalculateFactor_TOF(in_p, end_p, dif_time ,Reconpara.TOFinfo, Volpara, volume);
        //CalculateFactor_3D(in_p, end_p, Volpara, volume);

		total=extract_nonzero_entries3D_short(volume, Volpara.num[0], Volpara.num[1], Volpara.num[2], indices, entries);
		
		total_all+=total;
					
		fwrite(&total , 1 , sizeof(int) , fid);

		if(total)
		{

			fwrite(indices , total , sizeof(int) , fid );
			fwrite(entries , total , sizeof(short) , fid );		
		}

		/*
		for (ind=0;ind<total;ind++)
		{

			printf("INDICES[%d]=%d\n",ind,indices[ind]);
			printf("entries[%d]=%d\n",ind,entries[ind]);
		}
		*/




		sm_index++;
/****************************************************************************/
							
		free(indices);
		free(entries);	
		
	}
	fclose(fid_ls);

	//total_all=846538046;
	fwrite(&total_all,1,sizeof(int), fid);
	printf("total_all=%d\n",total_all);
	fclose(fid);
	free3Dfmatrix(volume,Volpara.num[0],Volpara.num[1],Volpara.num[2]);
}


