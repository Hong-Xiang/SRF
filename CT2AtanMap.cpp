#include "recon_req.h"

#define U_water 0.096 // Linear Attenuation Coefficient (LAC) of water at 511 KeV (unit: /cm)
#define UM_water 0.096// Mass Attenuation Coefficient (MAC) of water at 511 KeV (unit: cm^2/g)
#define UM_cb 0.092 //Mass Attenuation Coefficient (MAC) of cortical bone at 511 KeV (unit: cm^2/g)
#define Den_cb 1.88 // density of cortical bone (unit: g/cm^3)

int Vol2CT (int Voltage, char database_fname[128]);

void main()
{
	char CTfname[128];
	char database_fname[128];
	FILE *fid;
	short *CTimage; 
	int Nvolume;
	int Voltage;
	int ind;
	float *u_map;
	int HU_cb;


	Nvolume=128*128*60;
	Voltage=150;

	//strcat(CTfname,"");
	strcpy(database_fname,"E:\\F\\PET_reconstruction\\codes\\HU.txt");
	CTimage=sivector(Nvolume);

	u_map=fvector(Nvolume);


	if((fid = fopen(CTfname, "rb"))==NULL)
	{ 
		fprintf(stderr, "Could not open CT image file \"%s\" for reading\n", CTfname);
		exit(1); 
	}

	fread(CTimage,Nvolume, sizeof(int), fid);
	fclose(fid);
	
	for (ind=0;ind<Nvolume;ind++)
	{
		if (CTimage[ind]<=0)
		{
			u_map[ind]=(1+CTimage[ind]*0.001)*U_water;
		}
		else
		{	
			HU_cb=Vol2CT(Voltage,database_fname); 
			if (HU_cb==-1)
			{
				fprintf(stderr, "Could not find the HU corresponding to Voltage \"%d\" \n", Voltage);
				exit(1);
			}
			u_map[ind]=(1+(Den_cb*(UM_cb/(UM_water+realmin))-1)*CTimage[ind]/(HU_cb+realmin))*U_water;
		}
	}
}


int Vol2CT (int Voltage, char database_fname[128])
{
	FILE *fid;
	int database_Voltage;
	int database_HU;
	char Voltage_label[128];
	char HU_label[128];
	int HU=-1;

	if((fid = fopen(database_fname, "r"))==NULL)
	{ 
		fprintf(stderr, "Could not open database file \"%s\" for reading\n", database_fname);
		exit(1); 
	}
	fscanf(fid,"%s%s",&Voltage_label, &HU_label);

	while (fscanf(fid, "%d%d",&database_Voltage, &database_HU)!=EOF)
	{
		if (database_Voltage==Voltage)
		{
			HU=database_HU;
			break;
		}

	}
	fclose(fid);
	return HU;
}