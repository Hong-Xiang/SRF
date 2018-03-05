
#include "pet_req.h"





double max(double a, double b)
{
	double tmp;
	if (a>b) 
		tmp=a;
	else
		tmp=b;

	return tmp;
}

double min(double a, double b)
{
	double tmp;

	 if(a>b)
		 tmp=b;
	 else
		 tmp=a;

	 return tmp;
}

void BublleSort (double *arr, int count)
{
     int i, j;
	 int ind;
	 double temp;


	double *arr_tmp;


	arr_tmp =(double *) calloc(count,sizeof(double));

	for (ind=0;ind<count;ind++)
	{
		arr_tmp[ind]=arr[ind];
	}

	 for(j=0; j<count-1;j++) /* 冒泡法要排序n-1次*/
     {
		 for(i=0; i<count-j-1;i++)/* 值比较大的元素沉下去后，只把剩下的元素中的最大值再沉下去就可以啦 */
         {
                if(arr_tmp[i]>arr_tmp[i+1])/* 把值比较大的元素沉到底 */
                {
                    temp=arr_tmp[i+1];
                    arr_tmp[i+1]=arr_tmp[i];
                    arr_tmp[i]=temp;
                }
		 }
	 }

	 for (ind=0;ind<count;ind++)
	{
		arr[ind]=arr_tmp[ind];
	}
	free(arr_tmp);
}

int extract_nonzero_entries3D_short(float ***volume, int Nx, int Ny, int Nz, int *indices, short *entries)
{
  int nx, ny, nz, total=0;
  
  for(nz=0; nz<Nz; nz++)
    for(ny=0; ny<Ny; ny++)
      for(nx=0; nx<Nx; nx++)
        if(volume[nz][ny][nx] > 0.0039)
        {
          indices[total] = nx + Nx*(ny+Ny*nz);
          entries[total] = short(volume[nz][ny][nx]*255);
          total++;
        }
  return(total);
}

int extract_nonzero_entries3D(float ***volume, int Nx, int Ny, int Nz, int *indices, float *entries)
{
  int nx, ny, nz, total=0;
  
  for(nz=0; nz<Nz; nz++)
    for(ny=0; ny<Ny; ny++)
      for(nx=0; nx<Nx; nx++)
        if(volume[nz][ny][nx] > DBL_MIN)
        {
          indices[total] = nx + Nx*(ny+Ny*nz);
          entries[total] = volume[nz][ny][nx];
          total++;
        }
  return(total);
}

int extract_nonzero_entries2D(float **volume, int Nx, int Ny, int *indices, float *entries)
{
  int nx, ny, total=0;
  
    for(ny=0; ny<Ny; ny++)
      for(nx=0; nx<Nx; nx++)
        if(volume[ny][nx] > DBL_MIN)
        {
          indices[total] = nx + Nx*ny;
          entries[total] = volume[ny][nx];
          total++;
        }
  return(total);
}

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}

float *fvector(long nxl)
/* allocate a 1-D float vector with subscript range m[0..nxl] */
{
	float *m;

	m =(float *) calloc(nxl, sizeof(float));

	return m;
}

int *ivector(long nxl)
/* allocate a 1-D int vector with subscript range m[0..nxl] */
{
	int *m;

	m =(int *) calloc(nxl, sizeof(int));

	return m;
}

short *sivector(long nxl)
/* allocate a 1-D short vector with subscript range m[0..nxl] */
{
	short *m;

	m =(short *) calloc(nxl, sizeof(short));

	return m;
}

float **f2Dmatrix(long nxl,long nyl)
/* allocate a 2-D float matrix with subscript range m[0..nyl][0..nxl] */
{
	int ind;
	float **m;

	m =(float **) malloc(sizeof(float *) *nyl);
	for (ind=0;ind<nyl;ind++)
		m[ind]= (float *) calloc(nxl, sizeof(float));

	return m;
}

int **i2Dmatrix(long nxl,long nyl)
/* allocate a 2-D int matrix with subscript range m[0..nyl][0..nxl] */
{
	int ind;
	int **m;

	m =(int **) malloc(sizeof(int *) *nyl);

	for (ind=0;ind<nyl;ind++)
		m[ind]= (int *) calloc(nxl, sizeof(int));

	return m;
}

short **si2Dmatrix(long nxl,long nyl)
/* allocate a 2-D short matrix with subscript range m[0..nyl][0..nxl] */
{
	int ind;
	short **m;

	m =(short **) malloc(sizeof(short *) *nyl);

	for (ind=0;ind<nyl;ind++)
		m[ind]= (short *) calloc(nxl, sizeof(short));

	return m;
}

float ***f3Dmatrix(long nxl,long nyl, long nzl)
/* allocate a 3-D float matrix with subscript range m[0..nzl][0..nyl][0..nxl] */
{
	int ind_1,ind_2;
	float ***m;

	m=(float ***) malloc(sizeof(float **) *nzl);

	for (ind_1=0;ind_1<nzl;ind_1++)
	{	
		m[ind_1]= (float **) malloc(sizeof(float *) * nyl);
		for (ind_2=0;ind_2<nyl;ind_2++)
		{
			m[ind_1][ind_2]= (float *) calloc(nxl, sizeof(float));
		}
	}

	return m;
}

int ***i3Dmatrix(long nxl,long nyl, long nzl)
/* allocate a 3-D int matrix with subscript range m[0..nzl][0..nyl][0..nxl] */
{
	int ind_1,ind_2;
	int ***m;

	m=(int ***) malloc(sizeof(int **) *nzl);

	for (ind_1=0;ind_1<nzl;ind_1++)
	{	
		m[ind_1]= (int **) malloc(sizeof(int *) * nyl);

		for (ind_2=0;ind_2<nyl;ind_2++)
		{
			m[ind_1][ind_2]= (int *) calloc(nxl, sizeof(int));
		}
	}

	return m;
}

float ****f4Dmatrix(long nxl,long nyl, long nzl, long time)
/* allocate a 4-D float matrix with subscript range m[0..time][0..nzl][0..nyl][0..nxl] */
{
	int ind_0, ind_1,ind_2;
	float ****m;

	m=(float ****) malloc(sizeof(float***) *time);

	for (ind_0=0;ind_0<time;ind_0++)
	{
		m[ind_0]=(float ***) malloc(sizeof(float **) *nzl);

		for (ind_1=0;ind_1<nzl;ind_1++)
		{	
			m[ind_0][ind_1]= (float **) malloc(sizeof(float *) * nyl);
			for (ind_2=0;ind_2<nyl;ind_2++)
			{
				m[ind_0][ind_1][ind_2]= (float *) calloc(nxl, sizeof(float));
			}
		}
	}

	return m;
}


void free2Dfmatrix(float **m, long nxl,long nyl)
/* free a 2-D float matrix with subscript range m[0..nyl][0..nxl] */
{
	int ind;

	for (ind=0;ind<nyl;ind++) free(m[ind]);
	free(m);
}

void free3Dfmatrix(float ***m, long nxl,long nyl, long nzl)
/* free a 3-D float matrix with subscript range m[0..nzl][0..nyl][0..nxl] */
{
	int ind_1,ind_2;
	
	for (ind_1=0;ind_1<nzl;ind_1++)
	{
		for (ind_2=0;ind_2<nyl;ind_2++) 
		{
			free(m[ind_1][ind_2]);
		}
		free(m[ind_1]);
	}
	free(m);
}

void free3Dimatrix(int ***m, long nxl,long nyl, long nzl)
/* free a 3-D float matrix with subscript range m[0..nzl][0..nyl][0..nxl] */
{
	int ind_1,ind_2;
	
	for (ind_1=0;ind_1<nzl;ind_1++)
	{
		for (ind_2=0;ind_2<nyl;ind_2++) 
		{
			free(m[ind_1][ind_2]);
		}
		free(m[ind_1]);
	}
	free(m);
}

void free4Dfmatrix(float ****m, long nxl,long nyl, long nzl, long time)
/* free a 4-D float matrix with subscript range m[0..time][0..nzl][0..nyl][0..nxl] */
{
	int ind_0, ind_1,ind_2;
	
	for (ind_0=0;ind_0<time;ind_0++)
	{
		for (ind_1=0;ind_1<nzl;ind_1++)
		{
			for (ind_2=0;ind_2<nyl;ind_2++) 
			{
				free(m[ind_0][ind_1][ind_2]);
			}
			free(m[ind_0][ind_1]);
		}
		free(m[ind_0]);
	}
	free(m);
}


void f1Dassign(float *m, long nxl, float value)
{
	int ind_1;
	for (ind_1=0;ind_1<nxl;ind_1++)
	{
		m[ind_1]=value;
	}

}

void f2Dassign(float **m, long nxl,long nyl, float value)
{
	int ind_1,ind_2;
	for (ind_1=0;ind_1<nyl;ind_1++)
	{
		for (ind_2=0;ind_2<nxl;ind_2++)
		{
			m[ind_1][ind_2]=value;
		}
	}
}

void f3Dassign(float ***m, long nxl,long nyl, long nzl, float value)
{
	int ind_1,ind_2,ind_3;
	for (ind_1=0;ind_1<nzl;ind_1++)
	{
		for (ind_2=0;ind_2<nyl;ind_2++)
		{
			for (ind_3=0;ind_3<nxl;ind_3++)
			{
				m[ind_1][ind_2][ind_3]=value;
			}
		}
	}
}

void i3Dassign(int ***m, long nxl,long nyl, long nzl, int value)
{
	int ind_1,ind_2,ind_3;
	for (ind_1=0;ind_1<nzl;ind_1++)
	{
		for (ind_2=0;ind_2<nyl;ind_2++)
		{
			for (ind_3=0;ind_3<nxl;ind_3++)
			{
				m[ind_1][ind_2][ind_3]=value;
			}
		}
	}
}
void f2Dwrite(float **m,long nxl, long nyl, FILE *fid)
{
	int ind_0; 

	for (ind_0=0;ind_0<nyl;ind_0++)
	{	
			fwrite(m[ind_0],sizeof(float),nxl, fid);
	}
}

void f3Dwrite(float ***m,long nxl, long nyl, long nzl, FILE *fid)
{
	int ind_0,ind_1; 
	
	for (ind_0=0;ind_0<nzl;ind_0++)
	{
		for (ind_1=0;ind_1<nyl;ind_1++)
		{	
			fwrite(m[ind_1],sizeof(float),nxl, fid);
		}
	}
}


void i3Dwrite(int ***m,long nxl, long nyl, long nzl, FILE *fid)
{
	int ind_0,ind_1; 
	
	for (ind_0=0;ind_0<nzl;ind_0++)
	{
		for (ind_1=0;ind_1<nyl;ind_1++)
		{	
			fwrite(m[ind_1],sizeof(int),nxl, fid);
		}
	}
}

