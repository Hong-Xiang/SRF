#include "recon_req.h"


int Comimglocation(int oldloc, int N[3], int z_ind, int sym_ind, int SYM_Z)
{
	int newloc;
	int nx,ny,nz;
	int new_nx,new_ny,new_nz;

	nz=oldloc/(N[0]*N[1]);
	

	ny=(oldloc%(N[0]*N[1]))/N[1];
	nx=(oldloc%(N[0]*N[1]))%N[1];

	new_nz=nz+z_ind*SYM_Z;
/**********************SYM=2**************************/
/*
	switch 	(sym_ind) 
	{
		case 0:
			new_ny=ny;
			new_nx=nx;
			break;
		case 1:
			new_ny=nx;
			new_nx=N[1]-1-ny;
			break;

		default:
			fprintf(stderr,"something is wrong with the parameters of the symmetry!");
			break; 
		
	}
/****************************************************/
/**********************SYM=4*************************/
	switch 	(sym_ind) 
	{
		case 0:
			new_ny=ny;
			new_nx=nx;
			break;
		case 1:
			new_ny=nx;
			new_nx=ny;
			break;
		case 2:
			new_ny=nx;
			new_nx=N[1]-1-ny;
			break;
		case 3:
			new_ny=ny;
			new_nx=N[1]-1-nx;
			break;
		case 4:
			new_ny=N[1]-1-ny;
			new_nx=N[1]-1-nx;
			break;
		default:
			fprintf(stderr,"something is wrong with the parameters of the symmetry!");
			break; 
		
	}
/****************************************************/
/***********************SYM=8************************/
	/*
	switch 	(sym_ind) 
	{
		case 0:
			new_ny=ny;
			new_nx=nx;
			break;
		case 1:
			new_ny=nx;
			new_nx=ny;
			break;
		case 2:
			new_ny=nx;
			new_nx=N[1]-1-ny;
			break;
		case 3:
			new_ny=ny;
			new_nx=N[1]-1-nx;
			break;
		case 4:
			new_ny=N[1]-1-ny;
			new_nx=N[1]-1-nx;
			break;
		case 5:
			new_ny=N[1]-1-nx;
			new_nx=N[1]-1-ny;
			break;
		case 6:
			new_ny=N[1]-1-nx;
			new_nx=ny;
			break;
		case 7:
			new_ny=N[1]-1-ny;
			new_nx=nx;
			break;
		default:
			fprintf(stderr,"something is wrong with the parameters of the symmetry!");
			break; 
		
	}
/************************************************************************************/
	newloc=new_nz*N[0]*N[1]+new_ny*N[1]+new_nx;
	
	
	if(newloc>=(N[0]*N[1]*N[2]))
	{
		printf("new_nz=%d\n",new_nz);
		printf("new_ny=%d\n",new_ny);
		printf("new_nx=%d\n",new_nx);

		printf("old_nz=%d\n",nz);
		printf("old_ny=%d\n",ny);
		printf("old_nx=%d\n",nx);
	}
	
	return newloc;
}


int Comsinlocation(int oldloc, int Nsinogram, int SYM, int sym_ind, int z_ind, int Znuma)
{
	int newloc;
/******************************************SYM=2*******************************************/
/*	newloc=oldloc+Nsinogram*sym_ind+z_ind*(Nsinogram*SYM) +Znuma*(Nsinogram*SYM);

/******************************************************************************************/
/******************************************SYM=4********************************************/
	switch 	(sym_ind) 
	{
		case 0: case 2:
			newloc=oldloc+Nsinogram*sym_ind+z_ind*(Nsinogram*SYM) +Znuma*(Nsinogram*SYM);
			break;
		case 1: case 3:
			newloc=(Nsinogram-1-oldloc)+Nsinogram*sym_ind+z_ind*(Nsinogram*SYM)+Znuma*(Nsinogram*SYM);
			break;
		default:
			fprintf(stderr,"something is wrong with the parameters of the symmetry!");
			break; 
	}
/*****************************************sym=8*********************************************/
	/*
	switch 	(sym_ind) 
	{
		case 0: case 2: case 4: case 6:
			newloc=oldloc+Nsinogram*sym_ind+z_ind*(Nsinogram*SYM) +Znuma*(Nsinogram*SYM);
			break;
		case 1: case 3: case 5: case 7:
			newloc=Nsinogram-1-oldloc+Nsinogram*sym_ind+z_ind*(Nsinogram*SYM)+Znuma*(Nsinogram*SYM);
			break;
		default:
			fprintf(stderr,"something is wrong with the parameters of the symmetry!");
			break; 
	}
/*************************************************************************************/
	return newloc;
}

void CalculateFactor_3D(double *sou_p, double *end_p, VolumeParameter VolPara, float ***volume)
	/*input parametrer:
						 coordinate in X,Y, Z axis of the start points of LOR: sou_p[3]
						 coordinate in X,Y, Z axis of the start points of LOR: end_p[3]
						 the struct to define the image volume( see the definition in header): VolPara
						 the 3-D image volume: volume
	*/
{
	
	double alphax0, alphaxn, alphay0, alphayn, alphaz0, alphazn;
	double alphaxmin, alphaxmax, alphaymin, alphaymax, alphazmin, alphazmax, alphac;
	double alphamin, alphamax, alphaavg, alphax, alphay, alphaz, alphaxu, alphayu, alphazu;
	double alphatemp;
	double phyx_alphamin, phyx_alphamax, phyy_alphamin, phyy_alphamax, phyz_alphamin, phyz_alphamax;

	double p1x, p1y, p1z, p2x, p2y, p2z, pdx, pdy, pdz;
	double dconv;

	int i_f, i_l, j_f, j_l, k_f, k_l, i_min, i_max, j_min, j_max, k_min, k_max, iu, ju, ku;
	int Nx, Ny, Nz, Np;
	int i, xindex, yindex, zindex;

	p1x = sou_p[0];
	p2x = end_p[0];
	pdx = p2x - p1x;

	p1y = sou_p[1];
	p2y = end_p[1];
	pdy = p2y - p1y;

	p1z = sou_p[2];
	p2z = end_p[2];
	pdz = p2z - p1z;

	Nx = VolPara.num[0]+1 ;
	Ny = VolPara.num[1]+1 ;
	Nz = VolPara.num[2]+1 ;

	alphax0 = (VolPara.org[0] - p1x) / (pdx + realmin);
	alphaxn = (VolPara.org[0] + (Nx - 1)*VolPara.detla[0] - p1x) / (pdx + realmin);

	alphay0 = (VolPara.org[1] - p1y) / (pdy + realmin);
	alphayn = (VolPara.org[1] + (Ny - 1)*VolPara.detla[1] - p1y) / (pdy + realmin);

	alphaz0 = (VolPara.org[2] - p1z) / (pdz + realmin);
	alphazn = (VolPara.org[2] + (Nz - 1)*VolPara.detla[2] - p1z) / (pdz + realmin);

	alphaxmin = min(alphax0, alphaxn);alphaxmax = max(alphax0, alphaxn);
	alphaymin = min(alphay0, alphayn);alphaymax = max(alphay0, alphayn);
	alphazmin = min(alphaz0, alphazn);alphazmax = max(alphaz0, alphazn);

	alphatemp = max(alphaxmin, alphaymin);
	alphamin = max(alphatemp, alphazmin);

	alphatemp = min(alphaxmax, alphaymax);
	alphamax = min(alphatemp, alphazmax);

	if (alphamin < alphamax)
	{
		phyx_alphamin = (p1x + alphamin*pdx - VolPara.org[0]) / VolPara.detla[0];
		phyx_alphamax = (p1x + alphamax*pdx - VolPara.org[0]) / VolPara.detla[0];

		phyy_alphamin = (p1y + alphamin*pdy - VolPara.org[1]) / VolPara.detla[1];
		phyy_alphamax = (p1y + alphamax*pdy - VolPara.org[1]) / VolPara.detla[1];

		phyz_alphamin = (p1z + alphamin*pdz - VolPara.org[2]) / VolPara.detla[2];
		phyz_alphamax = (p1z + alphamax*pdz - VolPara.org[2]) / VolPara.detla[2];


		if (p1x < p2x)
		{
			if (alphamin == alphaxmin)    i_f = 1;
			else                          i_f = ceil(phyx_alphamin);
			if (alphamax == alphaxmax)    i_l = Nx - 1;
			else                          i_l = floor(phyx_alphamax);
			iu = 1;
			alphax = (VolPara.org[0] + i_f*VolPara.detla[0] - p1x) / pdx;
		}
		else if (p1x > p2x)
		{
			if (alphamin == alphaxmin)    i_f = Nx - 2;
			else                          i_f = floor(phyx_alphamin);
			if (alphamax == alphaxmax)    i_l = 0;
			else                          i_l = ceil(phyx_alphamax);
			iu = -1;
			alphax = (VolPara.org[0] + i_f*VolPara.detla[0] - p1x) / pdx;
		}
		else
		{
			i_f = int(phyx_alphamin);
			i_l = int(phyx_alphamax);
			iu = 0;
			alphax = realmax;
		}

		if (p1y < p2y)
		{
			if (alphamin == alphaymin)    j_f = 1;
			else                          j_f = ceil(phyy_alphamin);
			if (alphamax == alphaymax)    j_l = Ny - 1;
			else                          j_l = floor(phyy_alphamax);
			ju = 1;
			alphay = (VolPara.org[1] + j_f*VolPara.detla[1] - p1y) / pdy;
		}
		else if (p1y > p2y)
		{
			if (alphamin == alphaymin)    j_f = Ny - 2;
			else                          j_f = floor(phyy_alphamin);
			if (alphamax == alphaymax)    j_l = 0;
			else                          j_l = ceil(phyy_alphamax);
			ju = -1;
			alphay = (VolPara.org[1] + j_f*VolPara.detla[1] - p1y) / pdy;
		}
		else
		{
			j_f = int(phyy_alphamin);
			j_l = int(phyy_alphamax);
			ju = 0;
			alphay = realmax;
		}


		if (p1z < p2z)
		{
			if (alphamin == alphazmin)    k_f = 1;
			else                          k_f = ceil(phyz_alphamin);
			if (alphamax == alphazmax)    k_l = Nz - 1;
			else                          k_l = floor(phyz_alphamax);
			ku = 1;
			alphaz = (VolPara.org[2] + k_f*VolPara.detla[2] - p1z) / pdz;
		}
		else if (p1z > p2z)
		{
			if (alphamin == alphazmin)    k_f = Nz - 2;
			else                          k_f = floor(phyz_alphamin);
			if (alphamax == alphazmax)    k_l = 0;
			else                          k_l = ceil(phyz_alphamax);
			ku = -1;
			alphaz = (VolPara.org[2] + k_f*VolPara.detla[2] - p1z) / pdz;
		}
		else
		{
			k_f = int(phyz_alphamin);
			k_l = int(phyz_alphamax);
			ku = 0;
			alphaz = realmax;
		}

		i_min = min(i_f, i_l);
		i_max = max(i_f, i_l);
		j_min = min(j_f, j_l);
		j_max = max(j_f, j_l);
		k_min = min(k_f, k_l);
		k_max = max(k_f, k_l);


		Np = (i_max - i_min + 1) + (j_max - j_min + 1) + (k_max - k_min + 1);

		alphatemp = min(alphax, alphay);
		alphaavg = (min(alphatemp, alphaz) + alphamin) / 2;

		xindex = int(((p1x + alphaavg*pdx) - VolPara.org[0]) / VolPara.detla[0]);
		yindex = int(((p1y + alphaavg*pdy) - VolPara.org[1]) / VolPara.detla[1]);
		zindex = int(((p1z + alphaavg*pdz) - VolPara.org[2]) / VolPara.detla[2]);

		alphaxu = VolPara.detla[0] / (fabs(pdx) + realmin);
		alphayu = VolPara.detla[1] / (fabs(pdy) + realmin);
		alphazu = VolPara.detla[2] / (fabs(pdz) + realmin);

		alphac = alphamin;
		dconv = sqrt(pdx*pdx + pdy*pdy + pdz*pdz);

		for (i = 0;i < Np;i++)
		{
			if (alphax < alphay && alphax < alphaz)
			{
				if (xindex >= 0 && xindex <= (Nx - 2) && yindex >= 0 && yindex <= (Ny - 2) && zindex >= 0 && zindex <= (Nz - 2))
				{
					volume[zindex][xindex][yindex]=(alphax - alphac)*dconv;
					xindex = xindex + iu;
					alphac = alphax;
					alphax = alphax + alphaxu;
				}
			}
			else if (alphay < alphaz)
			{
				if (xindex >= 0 && xindex <= (Nx - 2) && yindex >= 0 && yindex <= (Ny - 2) && zindex >= 0 && zindex <= (Nz - 2))
				{
					volume[zindex][xindex][yindex]=(alphay - alphac)*dconv;
					yindex = yindex + ju;
					alphac = alphay;
					alphay = alphay + alphayu;
				}
			}
			else
			{
				if (xindex >= 0 && xindex <= (Nx - 2) && yindex >= 0 && yindex <= (Ny - 2) && zindex >= 0 && zindex <= (Nz - 2))
				{
					volume[zindex][xindex][yindex]=(alphaz - alphac)*dconv;

					zindex = zindex + ku;
					alphac = alphaz;
					alphaz = alphaz + alphazu;
				}
			}
		}

	}
}

void CalculateFactor_2D(double *sou_p, double *end_p, VolumeParameter VolPara, float **volume)
		/*input parametrer:
						 coordinate in X,Y axis of the start points of LOR: sou_p[3]
						 coordinate in X,Y axis of the start points of LOR: end_p[3]
						 the struct to define the image volume( see the definition in header): VolPara
						 the 2-D image volume: volume
	*/
{
	
	double alphax0, alphaxn, alphay0, alphayn;
	double alphaxmin, alphaxmax, alphaymin, alphaymax, alphac;
	double alphamin, alphamax, alphaavg, alphax, alphay, alphaxu, alphayu;
//	double alphatemp;
	double phyx_alphamin, phyx_alphamax, phyy_alphamin, phyy_alphamax;

	double p1x, p1y, p2x, p2y, pdx, pdy;
	double dconv;

	int i_min, i_max, j_min, j_max, iu, ju;
	int i_f, i_l, j_f, j_l;
	int Nx, Ny, Np;
	int i, xindex, yindex;




	p1x = sou_p[0];
	p2x = end_p[0];
	pdx = p2x - p1x;

	p1y = sou_p[1];
	p2y = end_p[1];
	pdy = p2y - p1y;

	Nx = VolPara.num[0] +1;
	Ny = VolPara.num[1] +1;

	alphax0 = (VolPara.org[0] - p1x) / (pdx + realmin);
	alphaxn = (VolPara.org[0] + (Nx - 1)*VolPara.detla[0] - p1x) / (pdx + realmin);

	alphay0 = (VolPara.org[1] - p1y) / (pdy + realmin);
	alphayn = (VolPara.org[1] + (Ny - 1)*VolPara.detla[1] - p1y) / (pdy + realmin);

	alphaxmin = min(alphax0, alphaxn);alphaxmax = max(alphax0, alphaxn);
	alphaymin = min(alphay0, alphayn);alphaymax = max(alphay0, alphayn);


	alphamin = max(alphaxmin, alphaymin);	
	alphamax = min(alphaxmax, alphaymax);



	if (alphamin < alphamax)
	{
		phyx_alphamin = (p1x + alphamin*pdx - VolPara.org[0]) / VolPara.detla[0];
		phyx_alphamax = (p1x + alphamax*pdx - VolPara.org[0]) / VolPara.detla[0];

		phyy_alphamin = (p1y + alphamin*pdy - VolPara.org[1]) / VolPara.detla[1];
		phyy_alphamax = (p1y + alphamax*pdy - VolPara.org[1]) / VolPara.detla[1];


		if (p1x < p2x)
		{
			if (alphamin == alphaxmin)    i_f = 1;
			else                          i_f = int(phyx_alphamin)+1;
			if (alphamax == alphaxmax)    i_l = Nx - 1;
			else                          i_l = int(phyx_alphamax);
			iu = 1;
			alphax = (VolPara.org[0] + i_f*VolPara.detla[0] - p1x) / pdx;
		}
		else if (p1x > p2x)
		{
			if (alphamin == alphaxmin)    i_f = Nx - 2;
			else                          i_f = int(phyx_alphamin);
			if (alphamax == alphaxmax)    i_l = 0;
			else                          i_l = int(phyx_alphamax)+1;
			iu = -1;
			alphax = (VolPara.org[0] + i_f*VolPara.detla[0] - p1x) / pdx;
		}
		else
		{
			i_f = int(phyx_alphamin);
			i_l = int(phyx_alphamax);
			iu = 0;
			alphax = realmax;
		}

		if (p1y < p2y)
		{
			if (alphamin == alphaymin)    j_f = 1;
			else                          j_f = int(phyy_alphamin)+1;
			if (alphamax == alphaymax)    j_l = Ny - 1;
			else                          j_l = int(phyy_alphamax);
			ju = 1;
			alphay = (VolPara.org[1] + j_f*VolPara.detla[1] - p1y) / pdy;
		}
		else if (p1y > p2y)
		{
			if (alphamin == alphaymin)    j_f = Ny - 2;
			else                          j_f = int(phyy_alphamin);
			if (alphamax == alphaymax)    j_l = 0;
			else                          j_l = int(phyy_alphamax)+1;
			ju = -1;
			alphay = (VolPara.org[1] + j_f*VolPara.detla[1] - p1y) / pdy;
		}
		else
		{
			j_f = int(phyy_alphamin);
			j_l = int(phyy_alphamax);
			ju = 0;
			alphay = realmax;
		}

		i_min = min(i_f, i_l);i_max = max(i_f, i_l);
		j_min = min(j_f, j_l);j_max = max(j_f, j_l);


		Np = (i_max - i_min + 1) + (j_max - j_min + 1);

		alphaavg = (min(alphax, alphay) + alphamin) / 2;

		xindex = int(((p1x + alphaavg*pdx) - VolPara.org[0]) / VolPara.detla[0]);


		yindex = int(((p1y + alphaavg*pdy) - VolPara.org[1]) / VolPara.detla[1]);


	

		alphaxu = VolPara.detla[0] / (fabs(pdx) + realmin);
		alphayu = VolPara.detla[1] / (fabs(pdy) + realmin);

		alphac = alphamin;
		dconv = sqrt(pdx*pdx + pdy*pdy);

		for (i = 0;i < Np;i++)
		{
			if (alphax < alphay)
			{
				if (xindex >= 0 && xindex <= (Nx - 2) && yindex >= 0 && yindex <= (Ny - 2))
				{
					volume[xindex][yindex]=(alphax - alphac)*dconv;
					xindex = xindex + iu;
					alphac = alphax;
					alphax = alphax + alphaxu;
				}
			}
			else
			{
				if (xindex >= 0 && xindex <= (Nx - 2) && yindex >= 0 && yindex <= (Ny - 2))
				{
					volume[xindex][yindex]=(alphay - alphac)*dconv;
					yindex = yindex + ju;
					alphac = alphay;
					alphay = alphay + alphayu;
				}
			}
		}

	}

}


void CalculateFactor_siddon(double *in_p, double *end_p, VolumeParameter VolPara, float ***volume)
		/*input parametrer:
						 coordinate in X,Y, Z axis of the start points of LOR: sou_p[3]
						 coordinate in X,Y, Z axis of the start points of LOR: end_p[3]
						 the struct to define the image volume( see the definition in header): VolPara
						 the 3-D image volume: volume
	*/
{

	double dis_x_12,dis_y_12,dis_z_12; //the X/Y/Z distance between point 1 and point 2 
	double d_12;// the Euler distance between point 1 and point 2
	double d_x,d_y,d_z; //voxel size in x,y,z direction

	double *Xplane,*Yplane,*Zplane; //the coordinates of the intersected points
	double *alpha_x, *alpha_y, *alpha_z;// alpha values of the intersected points
	double alpha_x_0, alpha_y_0, alpha_z_0;
	double alpha_x_Nx_1, alpha_y_Ny_1, alpha_z_Nz_1;
	int Nx, Ny, Nz; // the numbers of the intersected points
	int ind_x, ind_y, ind_z;  //
	double alpha_x_min, alpha_y_min, alpha_z_min;  // the minimum values of the alpha in x/y/z direction
	double alpha_x_max, alpha_y_max, alpha_z_max;  // the maximum values of the alpha in x/y/z direction
	double alpha_tmp_1, alpha_tmp_2;  // the temp value of alpha
	double alpha_min, alpha_max;     // the minimum value of the alpha in x and y and z direction
	int i_min, j_min, k_min;       // the minimum value of the i/j/k value 
	int i_max, j_max, k_max;      // the maximum value of the i/j/k value  
	int ind_i,ind_j,ind_k,m;
	double *alpha;
	//double *alpha_1;  // the orginal alpha values and the sorted alpha values
	double alpha_mid;
//	int ind;
	int i, j, k;
	int n; // the number of intesected points of the line between point 1 and point 2;

	dis_x_12=end_p[0]-in_p[0];
	dis_y_12=end_p[1]-in_p[1];
	dis_z_12=end_p[2]-in_p[2];

	if(fabs(dis_x_12)<DBL_MIN) dis_x_12=DBL_MIN;
	if(fabs(dis_y_12)<DBL_MIN) dis_y_12=DBL_MIN;
	if(fabs(dis_z_12)<DBL_MIN) dis_z_12=DBL_MIN;

	Nx=VolPara.num[0]+1;
	Ny=VolPara.num[1]+1;
	Nz=VolPara.num[2]+1;

	d_x=VolPara.detla[0];
	d_y=VolPara.detla[1];
	d_z=VolPara.detla[2];


	Xplane =(double *) calloc(Nx, sizeof(double));
	Yplane =(double *) calloc(Ny, sizeof(double));
	Zplane =(double *) calloc(Nz, sizeof(double));

	
	Xplane[0]=VolPara.org[0];
	Yplane[0]=VolPara.org[1];
	Zplane[0]=VolPara.org[2];


		for (ind_x=1;ind_x<Nx;ind_x++)
	{
		Xplane[ind_x]=Xplane[0]+ind_x*d_x;
	}

	for (ind_y=1;ind_y<Ny;ind_y++)
	{
		Yplane[ind_y]=Yplane[0]+ind_y*d_y;
	}

	for (ind_z=1;ind_z<Nz;ind_z++)
	{
		Zplane[ind_z]=Zplane[0]+ind_z*d_z;
	}
/***************************************************************/
	alpha_x_0=(Xplane[0]-in_p[0])/dis_x_12;
	alpha_y_0=(Yplane[0]-in_p[1])/dis_y_12;
	alpha_z_0=(Zplane[0]-in_p[2])/dis_z_12;

	alpha_x_Nx_1=(Xplane[Nx-1]-in_p[0])/dis_x_12;
	alpha_y_Ny_1=(Yplane[Ny-1]-in_p[1])/dis_y_12;
	alpha_z_Nz_1=(Zplane[Nz-1]-in_p[2])/dis_z_12;

/****************************************************************/
	

	
	alpha_x_min=min(alpha_x_0,alpha_x_Nx_1);
	alpha_x_max=max(alpha_x_0,alpha_x_Nx_1);

	alpha_y_min=min(alpha_y_0,alpha_y_Ny_1);
	alpha_y_max=max(alpha_y_0,alpha_y_Ny_1);

	alpha_z_min=min(alpha_z_0,alpha_z_Nz_1);
	alpha_z_max=max(alpha_z_0,alpha_z_Nz_1);


	alpha_tmp_1=max(0,alpha_x_min);
	alpha_tmp_2=max(alpha_tmp_1,alpha_y_min);
	alpha_min=max(alpha_tmp_2,alpha_z_min);

	alpha_tmp_1=min(1,alpha_x_max);
	alpha_tmp_2=min(alpha_tmp_1,alpha_y_max);
	alpha_max=min(alpha_tmp_2,alpha_z_max);
/***************************************************************/


	if(alpha_max>alpha_min)
	{
		if (fabs(dis_x_12)<=DBL_MIN)
		{
			i_min=0;
			i_max=-1;
		}
		else
		{
			if (dis_x_12>0)
			{
				i_min=Nx-1-int((Xplane[Nx-1]-alpha_min*dis_x_12-in_p[0])/d_x);
				i_max=0+int((in_p[0]+alpha_max*dis_x_12-Xplane[0])/d_x);
			}
			else if (dis_x_12<0)
			{
				i_min=Nx-1-int((Xplane[Nx-1]-alpha_max*dis_x_12-in_p[0])/d_x);
				i_max=0+int((in_p[0]+alpha_min*dis_x_12-Xplane[0])/d_x);
			}
		}

		if (fabs(dis_y_12)<=DBL_MIN)
		{
			j_min=0;
			j_max=-1;
		}
		else
		{
			if (dis_y_12>0)
			{
				j_min=Ny-1-int((Yplane[Ny-1]-alpha_min*dis_y_12-in_p[1])/d_y);
				//printf("%f\n",in_p[1]+alpha_max*dis_y_12-Yplane[0]);
				j_max=0+int((in_p[1]+alpha_max*dis_y_12-Yplane[0])/d_y);
			}
			else if (dis_y_12<0)
			{
				j_min=Ny-1-int((Yplane[Ny-1]-alpha_max*dis_y_12-in_p[1])/d_y);
				j_max=0+int((in_p[1]+alpha_min*dis_y_12-Yplane[0])/d_y);
			}
		}

		if (fabs(dis_z_12)<=DBL_MIN)
		{
			k_min=0;
			k_max=-1;
		}
		else
		{
			if (dis_z_12>0)
			{
				k_min=Nz-1-int((Zplane[Nz-1]-alpha_min*dis_z_12-in_p[2])/d_z);
				k_max=0+int((in_p[2]+alpha_max*dis_z_12-Zplane[0])/d_z);
			}
			else if (dis_z_12<0)
			{
				k_min=Nz-1-int((Zplane[Nz-1]-alpha_max*dis_z_12-in_p[2])/d_z);
				k_max=0+int((in_p[2]+alpha_min*dis_z_12-Zplane[0])/d_z);
			}
		}

/***********************************************************************/


	alpha_x =(double *) calloc(i_max-i_min+1, sizeof(double));
	alpha_y =(double *) calloc(j_max-j_min+1, sizeof(double));
	alpha_z =(double *) calloc(k_max-k_min+1, sizeof(double));

	if (dis_x_12>0)
	{
		for (ind_i=0;ind_i<(i_max-i_min+1);ind_i++)
		{
			alpha_x[ind_i]=(Xplane[i_min+ind_i]-in_p[0])/dis_x_12;
		}
	}
	else
	{
		for (ind_i=0;ind_i<(i_max-i_min+1);ind_i++)
		{
			alpha_x[ind_i]=(Xplane[i_max-ind_i]-in_p[0])/dis_x_12;
		}
	}


	if(dis_y_12>0)
	{
		for(ind_j=0;ind_j<(j_max-j_min+1);ind_j++)
		{
			alpha_y[ind_j]=(Yplane[j_min+ind_j]-in_p[1])/dis_y_12;
		}
	}
	else
	{
		for(ind_j=0;ind_j<(j_max-j_min+1);ind_j++)
		{
			alpha_y[ind_j]=(Yplane[j_max-ind_j]-in_p[1])/dis_y_12;
		}
	}

	if(dis_z_12>0)
	{
		for(ind_k=0;ind_k<(k_max-k_min+1);ind_k++)
		{
			alpha_z[ind_k]=(Zplane[k_min+ind_k]-in_p[2])/dis_z_12;
		}
	}
	else
	{
		for(ind_k=0;ind_k<(k_max-k_min+1);ind_k++)
		{
			alpha_z[ind_k]=(Zplane[k_max-ind_k]-in_p[2])/dis_z_12;
		}
	}


	n=(i_max-i_min+1)+(j_max-j_min+1)+(k_max-k_min+1); //toal number of the intersected points + two end points (alpha_min & alpha_max)



	alpha =(double *) calloc(n, sizeof(double));
	
	for (ind_i=0;ind_i<(i_max-i_min+1);ind_i++)
	{
		alpha[ind_i]=alpha_x[ind_i];
	}

	for(ind_j=0;ind_j<(j_max-j_min+1);ind_j++)
	{
		alpha[ind_j+i_max-i_min+1]=alpha_y[ind_j];
	}

	for(ind_k=0;ind_k<(k_max-k_min+1);ind_k++)
	{
		alpha[ind_k+j_max-j_min+1+i_max-i_min+1]=alpha_z[ind_k];
	}

	

	BublleSort(alpha, n);


	d_12=sqrt(dis_x_12*dis_x_12+dis_y_12*dis_y_12+dis_z_12*dis_z_12);


	
	for (m=1;m<n;m++)
	{
		alpha_mid=(alpha[m]+alpha[m-1])/2;

		i=0+int((in_p[0]+alpha_mid*dis_x_12-Xplane[0])/d_x);
		j=0+int((in_p[1]+alpha_mid*dis_y_12-Yplane[0])/d_y);
		k=0+int((in_p[2]+alpha_mid*dis_z_12-Zplane[0])/d_z);


		volume[k][j][i]=d_12*(alpha[m]-alpha[m-1]);

	}

	free(alpha);
	free(alpha_x);
	free(alpha_y);
	free(alpha_z);

	}
	free(Xplane);
	free(Yplane);
	free(Zplane);


}


void Location2coord(System_info PET, int location, int ring_id, float delta_z, double *coord)
{
	int Blk_ind, Xtal_ind;
	float coord_1[3], coord_2[3];
	float delta_phi;
	float half_blk_size;
	float R ; // the distance from the center of system to center of xtals

	Blk_ind=location/PET.Xtal_blk;
	Xtal_ind=location%PET.Xtal_blk;
	delta_phi=2*PI/PET.Blk_num;
	R = PET.R;// +PET.Xtal_length / 2.0;

	half_blk_size=(PET.Xtal_size*PET.Xtal_blk+PET.Xtal_gap*(PET.Xtal_blk-1))/2.0;

	coord_1[0]=sin(Blk_ind*delta_phi-atan(half_blk_size/R))*sqrt(R*R+half_blk_size*half_blk_size);
	coord_1[1]=cos(Blk_ind*delta_phi-atan(half_blk_size/R))*sqrt(R*R+half_blk_size*half_blk_size);
	
	coord_2[0]=sin((Blk_ind+1)*delta_phi-atan(half_blk_size/R))*sqrt(R*R+half_blk_size*half_blk_size);
	coord_2[1]=cos((Blk_ind+1)*delta_phi-atan(half_blk_size/R))*sqrt(R*R+half_blk_size*half_blk_size);

	coord[0]=coord_1[0]+(coord_2[0]-coord_1[0])*Xtal_ind/PET.Xtal_blk;
	coord[1]=coord_1[1]+(coord_2[1]-coord_1[1])*Xtal_ind/PET.Xtal_blk;

	coord[2]=(ring_id+0.5-float(PET.Ring_num)/2)*delta_z;


}


void Location2coord_DOI(System_info PET, int location, int ring_id, int DOI_id, float delta_z, double *coord)
{
	int Blk_ind, Xtal_ind;
	float coord_1[3], coord_2[3];
	float delta_phi;
	float half_blk_size;
	float R; // the distance from the center of system to different xtals layers

	Blk_ind=location/PET.Xtal_blk;
	Xtal_ind=location%PET.Xtal_blk;
	delta_phi=2*PI/PET.Blk_num;

	R=PET.R+DOI_id*PET.DOI_info.DOI_reso+PET.DOI_info.DOI_reso/2.0;
	
	half_blk_size=(PET.Xtal_size*PET.Xtal_blk+PET.Xtal_gap*(PET.Xtal_blk-1))/2.0;
	
	coord_1[0]=sin(Blk_ind*delta_phi-atan(half_blk_size/R))*sqrt(R*R+half_blk_size*half_blk_size);
	coord_1[1]=cos(Blk_ind*delta_phi-atan(half_blk_size/R))*sqrt(R*R+half_blk_size*half_blk_size);
	
	coord_2[0]=sin((Blk_ind+1)*delta_phi-atan(half_blk_size/R))*sqrt(R*R+half_blk_size*half_blk_size);
	coord_2[1]=cos((Blk_ind+1)*delta_phi-atan(half_blk_size/R))*sqrt(R*R+half_blk_size*half_blk_size);

	coord[0]=coord_1[0]+(coord_2[0]-coord_1[0])*Xtal_ind/PET.Xtal_blk;
	coord[1]=coord_1[1]+(coord_2[1]-coord_1[1])*Xtal_ind/PET.Xtal_blk;

	coord[2]=(ring_id+0.5-float(PET.Ring_num)/2)*delta_z;
	

}


void CalculateFactor_TOF(double *sou_p, double *end_p, float TOF_dif_time, TOF_info TOFinfo, VolumeParameter VolPara, float ***volume)
	/*input parametrer:
						 coordinate in X,Y, Z axis of the start points of LOR: sou_p[3]
						 coordinate in X,Y, Z axis of the start points of LOR: end_p[3]
						 the struct to define the image volume( see the definition in header): VolPara
						 the 3-D image volume: volume
	*/
{
	
	float DETLA_TOF=10e-12*SPD;
	double alphax0, alphaxn, alphay0, alphayn, alphaz0, alphazn;
	double alphaxmin, alphaxmax, alphaymin, alphaymax, alphazmin, alphazmax, alphac;
	double alphamin, alphamax, alphaavg, alphax, alphay, alphaz, alphaxu, alphayu, alphazu;
	double alphatemp;
	double phyx_alphamin, phyx_alphamax, phyy_alphamin, phyy_alphamax, phyz_alphamin, phyz_alphamax;

	double p1x, p1y, p1z, p2x, p2y, p2z, pdx, pdy, pdz;
	double dconv;

	int i_f, i_l, j_f, j_l, k_f, k_l, i_min, i_max, j_min, j_max, k_min, k_max, iu, ju, ku;
	int Nx, Ny, Nz, Np;
	int i, xindex, yindex, zindex;

	float b,c;
	float TOF;
	float tof_t;


	p1x = sou_p[0];
	p2x = end_p[0];
	pdx = p2x - p1x;

	p1y = sou_p[1];
	p2y = end_p[1];
	pdy = p2y - p1y;

	p1z = sou_p[2];
	p2z = end_p[2];
	pdz = p2z - p1z;

	Nx = VolPara.num[0]+1 ;
	Ny = VolPara.num[1]+1 ;
	Nz = VolPara.num[2]+1 ;

	alphax0 = (VolPara.org[0] - p1x) / (pdx + realmin);
	alphaxn = (VolPara.org[0] + (Nx - 1)*VolPara.detla[0] - p1x) / (pdx + realmin);

	alphay0 = (VolPara.org[1] - p1y) / (pdy + realmin);
	alphayn = (VolPara.org[1] + (Ny - 1)*VolPara.detla[1] - p1y) / (pdy + realmin);

	alphaz0 = (VolPara.org[2] - p1z) / (pdz + realmin);
	alphazn = (VolPara.org[2] + (Nz - 1)*VolPara.detla[2] - p1z) / (pdz + realmin);

	alphaxmin = min(alphax0, alphaxn);alphaxmax = max(alphax0, alphaxn);
	alphaymin = min(alphay0, alphayn);alphaymax = max(alphay0, alphayn);
	alphazmin = min(alphaz0, alphazn);alphazmax = max(alphaz0, alphazn);

	alphatemp = max(alphaxmin, alphaymin);
	alphamin = max(alphatemp, alphazmin);

	alphatemp = min(alphaxmax, alphaymax);
	alphamax = min(alphatemp, alphazmax);

	if (alphamin < alphamax)
	{
		phyx_alphamin = (p1x + alphamin*pdx - VolPara.org[0]) / VolPara.detla[0];
		phyx_alphamax = (p1x + alphamax*pdx - VolPara.org[0]) / VolPara.detla[0];

		phyy_alphamin = (p1y + alphamin*pdy - VolPara.org[1]) / VolPara.detla[1];
		phyy_alphamax = (p1y + alphamax*pdy - VolPara.org[1]) / VolPara.detla[1];

		phyz_alphamin = (p1z + alphamin*pdz - VolPara.org[2]) / VolPara.detla[2];
		phyz_alphamax = (p1z + alphamax*pdz - VolPara.org[2]) / VolPara.detla[2];


		if (p1x < p2x)
		{
			if (alphamin == alphaxmin)    i_f = 1;
			else                          i_f = ceil(phyx_alphamin);
			if (alphamax == alphaxmax)    i_l = Nx - 1;
			else                          i_l = floor(phyx_alphamax);
			iu = 1;
			alphax = (VolPara.org[0] + i_f*VolPara.detla[0] - p1x) / pdx;
		}
		else if (p1x > p2x)
		{
			if (alphamin == alphaxmin)    i_f = Nx - 2;
			else                          i_f = floor(phyx_alphamin);
			if (alphamax == alphaxmax)    i_l = 0;
			else                          i_l = ceil(phyx_alphamax);
			iu = -1;
			alphax = (VolPara.org[0] + i_f*VolPara.detla[0] - p1x) / pdx;
		}
		else
		{
			i_f = int(phyx_alphamin);
			i_l = int(phyx_alphamax);
			iu = 0;
			alphax = realmax;
		}

		if (p1y < p2y)
		{
			if (alphamin == alphaymin)    j_f = 1;
			else                          j_f = ceil(phyy_alphamin);
			if (alphamax == alphaymax)    j_l = Ny - 1;
			else                          j_l = floor(phyy_alphamax);
			ju = 1;
			alphay = (VolPara.org[1] + j_f*VolPara.detla[1] - p1y) / pdy;
		}
		else if (p1y > p2y)
		{
			if (alphamin == alphaymin)    j_f = Ny - 2;
			else                          j_f = floor(phyy_alphamin);
			if (alphamax == alphaymax)    j_l = 0;
			else                          j_l = ceil(phyy_alphamax);
			ju = -1;
			alphay = (VolPara.org[1] + j_f*VolPara.detla[1] - p1y) / pdy;
		}
		else
		{
			j_f = int(phyy_alphamin);
			j_l = int(phyy_alphamax);
			ju = 0;
			alphay = realmax;
		}


		if (p1z < p2z)
		{
			if (alphamin == alphazmin)    k_f = 1;
			else                          k_f = ceil(phyz_alphamin);
			if (alphamax == alphazmax)    k_l = Nz - 1;
			else                          k_l = floor(phyz_alphamax);
			ku = 1;
			alphaz = (VolPara.org[2] + k_f*VolPara.detla[2] - p1z) / pdz;
		}
		else if (p1z > p2z)
		{
			if (alphamin == alphazmin)    k_f = Nz - 2;
			else                          k_f = floor(phyz_alphamin);
			if (alphamax == alphazmax)    k_l = 0;
			else                          k_l = ceil(phyz_alphamax);
			ku = -1;
			alphaz = (VolPara.org[2] + k_f*VolPara.detla[2] - p1z) / pdz;
		}
		else
		{
			k_f = int(phyz_alphamin);
			k_l = int(phyz_alphamax);
			ku = 0;
			alphaz = realmax;
		}

		i_min = min(i_f, i_l);
		i_max = max(i_f, i_l);
		j_min = min(j_f, j_l);
		j_max = max(j_f, j_l);
		k_min = min(k_f, k_l);
		k_max = max(k_f, k_l);


		Np = (i_max - i_min + 1) + (j_max - j_min + 1) + (k_max - k_min + 1);

		alphatemp = min(alphax, alphay);
		alphaavg = (min(alphatemp, alphaz) + alphamin) / 2;

		xindex = int(((p1x + alphaavg*pdx) - VolPara.org[0]) / VolPara.detla[0]);
		yindex = int(((p1y + alphaavg*pdy) - VolPara.org[1]) / VolPara.detla[1]);
		zindex = int(((p1z + alphaavg*pdz) - VolPara.org[2]) / VolPara.detla[2]);

		alphaxu = VolPara.detla[0] / (fabs(pdx) + realmin);
		alphayu = VolPara.detla[1] / (fabs(pdy) + realmin);
		alphazu = VolPara.detla[2] / (fabs(pdz) + realmin);

		alphac = alphamin;
		dconv = sqrt(pdx*pdx + pdy*pdy + pdz*pdz);

		for (i = 0;i < Np;i++)
		{
			if (alphax < alphay && alphax < alphaz)
			{
				if (xindex >= 0 && xindex <= (Nx - 2) && yindex >= 0 && yindex <= (Ny - 2) && zindex >= 0 && zindex <= (Nz - 2))
				{
					tof_t=((alphax + alphac)*dconv/2.0-dconv/2.0)/DETLA_TOF;
					c = (TOFinfo.Time_reso*SPD/2/2/sqrt(2*log(2.0)))/DETLA_TOF;
					b = (TOF_dif_time*SPD)/DETLA_TOF;
					
					if ((tof_t>(b-3*c))&(tof_t<(b+3*c)))
					TOF= exp(-((tof_t-b)*(tof_t-b))/(2*c*c));
					else
					TOF=0;

					volume[zindex][xindex][yindex]=(alphax - alphac)*dconv*TOF;
					xindex = xindex + iu;
					alphac = alphax;
					alphax = alphax + alphaxu;
				}
			}
			else if (alphay < alphaz)
			{
				if (xindex >= 0 && xindex <= (Nx - 2) && yindex >= 0 && yindex <= (Ny - 2) && zindex >= 0 && zindex <= (Nz - 2))
				{
					tof_t=((alphax + alphac)*dconv/2.0-dconv/2.0)/DETLA_TOF;
					c = (TOFinfo.Time_reso*SPD/2/2/sqrt(2*log(2.0)))/DETLA_TOF;
					b = (TOF_dif_time*SPD)/DETLA_TOF;
					
					if ((tof_t>(b-3*c))&(tof_t<(b+3*c)))
					TOF= exp(-((tof_t-b)*(tof_t-b))/(2*c*c));
					else
					TOF=0;


					volume[zindex][xindex][yindex]=(alphay - alphac)*dconv*TOF;
					yindex = yindex + ju;
					alphac = alphay;
					alphay = alphay + alphayu;
				}
			}
			else
			{
				if (xindex >= 0 && xindex <= (Nx - 2) && yindex >= 0 && yindex <= (Ny - 2) && zindex >= 0 && zindex <= (Nz - 2))
				{

					tof_t=((alphax + alphac)*dconv/2.0-dconv/2.0)/DETLA_TOF;
					c = (TOFinfo.Time_reso*SPD/2/2/sqrt(2*log(2.0)))/DETLA_TOF;
					b = (TOF_dif_time*SPD)/DETLA_TOF;

					if ((tof_t>(b-3*c))&(tof_t<(b+3*c)))
					TOF= exp(-((tof_t-b)*(tof_t-b))/(2*c*c));
					else
					TOF=0;

					volume[zindex][xindex][yindex]=(alphaz - alphac)*dconv*TOF;

					zindex = zindex + ku;
					alphac = alphaz;
					alphaz = alphaz + alphazu;
				}
			}
		}

	}
}



void list2sino(System_info PET, ReconParameter Reconpara, SinogramParameter Sinopara, char listfile_name[128], short *sinogram)
{
	FILE *fid; // file pointer for reading and writing
	int Coin_Num; // the number of the concidences in the listmode data
	int index_coin; // index of the concidence 
	int ind, seg_ind;
	int Ringnum_1; // the ring index for the first photon in each concidence
	int	Ringnum_2; // the ring index for the second photon in each cocidence
	int Ringnum_tmp;
	int Location_1; // the location index per ring for the first photon in each concidence
	int Location_2; // the location index per ring for the second photon in each concidence
	int Location_tmp;

	int Ringnum_diff; // the ring index difference between the two photon in one concidence 
	int Location_diff; //the location index difference between the two photon in one concidence
	int Segment_Num; // the segment index of the concidence
	int OS_Num; // the slice index of the concidence
	int mid_se; //
/*************************************************************************************************************/	

	float diff_time;
	int Phi_Num;
	int Dis_Num;
	int diff1,diff2,sigma;
	int Znum=0;
	int initial_numz=0;
	int *Znuma;


	Znuma= (int *) calloc(Reconpara.Nseg, sizeof(int));
	for (seg_ind=0;seg_ind<Reconpara.Nseg;seg_ind++)
	{
		initial_numz=max(0, (seg_ind+1)/2);	
		Znuma[seg_ind]=Znum;
		Znum+=PET.Ring_num-initial_numz;			
	}

/*************************************************************************************************************/
	
		

		if((fid = fopen(listfile_name, "rb"))==NULL)
		{	 
			fprintf(stderr, "Could not open listmode file \"%s\" for reading\n", listfile_name);
			exit(1); 
		}
		fread(&Coin_Num, sizeof(int), 1, fid);

		printf(" Transform %d list-mode data to sinograms \n",Coin_Num);

		for (ind=0;ind<Coin_Num;ind++)
		{	
			fread(&index_coin, sizeof(int), 1, fid);
			
			if(ind%(Coin_Num/10)==0)
			printf(" %d%% Completed\n",ind/(Coin_Num/10)*10);

			if(index_coin!=ind)
			{
				fprintf(stderr, "Read in coincidence index %d not equal to expected %d\n", index_coin, ind);
				exit(-1);
			}

			fread(&Ringnum_1, sizeof(int), 1, fid);
			fread(&Ringnum_2, sizeof(int), 1, fid);
			fread(&Location_1, sizeof(int), 1, fid);
			fread(&Location_2, sizeof(int), 1, fid);
			fread(&diff_time, sizeof(float), 1, fid);
			
			if (Location_1>Location_2)
			{
				Location_tmp=Location_1;
				Location_1=Location_2;
				Location_2=Location_tmp;
				
				Ringnum_tmp=Ringnum_1;
				Ringnum_1=Ringnum_2;
				Ringnum_2=Ringnum_tmp;
			}

			Ringnum_diff=abs(Ringnum_1-Ringnum_2);
			Location_diff=Location_2-Location_1;					
/*****************************calculate the slice index of the coincidence**************************************/
			mid_se=Ringnum_diff+1; //mid_se=((Ringnum_diff+((SPAN-1)/2))/SPAN)+1;

			if (mid_se==1)
			{
				Segment_Num=0;
				OS_Num=Znuma[Segment_Num]+Ringnum_1;
			}	
			else if ((mid_se>1)&((Ringnum_1-Ringnum_2)>0))
			{
				Segment_Num=(mid_se-1)*2;
				OS_Num=Znuma[Segment_Num]+Ringnum_2;
			}	
			else if ((mid_se>1)&((Ringnum_1-Ringnum_2)<0))
			{
				Segment_Num=(mid_se-1)*2-1;
				OS_Num=Znuma[Segment_Num]+Ringnum_1;
			}

/********************************************************************************************************************/
			Phi_Num=((Location_1+Location_2+(PET.Xtal_num/2)+1)/2)%(PET.Xtal_num/2);
/********************************************************************************************************************/
			if (abs(Location_1 - Phi_Num) < abs(Location_1 - (Phi_Num + PET.Xtal_num)))
				diff1=Location_1-Phi_Num;
			else
				diff1=Location_1-(Phi_Num+PET.Xtal_num);

			if (abs(Location_2 - Phi_Num) < abs(Location_2 - (Phi_Num + PET.Xtal_num)))
				diff2=Location_2-Phi_Num;
			else
				diff2=Location_2-(Phi_Num+PET.Xtal_num);

			if (abs(diff1)<abs(diff2))
				sigma=Location_1-Location_2;
			else
				sigma=Location_2-Location_1;

			if (sigma<0) sigma+=PET.Xtal_num;

			Dis_Num=sigma+Sinopara.dis/2-PET.Xtal_num/2;

/*****************************************************************************************************************/
			if ((Dis_Num<Sinopara.dis)&(Dis_Num>=0)&(Phi_Num<Sinopara.angle)&(Phi_Num>=0)&(OS_Num<Sinopara.slice)&(OS_Num>=0))
			{
				sinogram[OS_Num*Sinopara.angle*Sinopara.dis+Phi_Num*Sinopara.dis+Dis_Num]+=1;
			}
		}
		fclose(fid);

		return;
}




