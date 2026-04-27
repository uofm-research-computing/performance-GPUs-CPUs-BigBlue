//Simple openmp parallel single precision Jacobi iteration solver for a 2 dimensional heat equation
//This also updates the field by copying the new field into the old one.
#define PRECISION float
#include "common.h"
#include <omp.h>

#define THRESHOLD 0.00001
#define MAX_ITER 1000000

int main(int argc, char **argv)
{
	//Check that we have enough options
	if(argc!=4)
	{
		printf("Usage: %s nx ny seed\n",argv[0]);
		return 0;
	}

	//Read the options
	int nx, ny, seed;
	sscanf(argv[1],"%d",&nx);
	sscanf(argv[2],"%d",&ny);
	sscanf(argv[3],"%d",&seed);
	
	//Seed the random number generator for initial conditions
	srand(seed);
	
	//Allocate some memory for fields
	PRECISION *field=(PRECISION*)malloc(nx*ny*sizeof(PRECISION));
	PRECISION *nextfield=(PRECISION*)malloc(nx*ny*sizeof(PRECISION));

	//Initialize the field with noise
	for(int i=0;i<nx*ny;i++)
		field[i]=(PRECISION)rand()/(PRECISION)RAND_MAX;
	
	//Setup boundary conditions
	struct neigh boundary;
	boundary.n=1;
	boundary.s=1;
	boundary.e=0;
	boundary.w=0;

	//Step until maximum error is smaller than a threshold
	PRECISION error;
	int iter=0;
	do
	{
		error=0;
		#pragma omp parallel for reduction(max:error)
		for(int j=0;j<ny;j++)
		{
			for(int i=0;i<nx;i++)
			{
				int current=flattenCoord(i,j,nx,ny);
				struct neigh n=getNeighbors(i,j,nx,ny,field,boundary);
				nextfield[current]=fdm(n);
				PRECISION derror=fabs(nextfield[current]-field[current]);
				error=fmaxf(error,derror);
			}
		}
		PRECISION *buf=field;
		field=nextfield;
		nextfield=buf;
		iter++;
	} while(error>THRESHOLD && iter<MAX_ITER);
	
	//Output a CSV formatted string
	for(int i=0;i<nx;i++)
	{
		for(int j=0;j<ny;j++)
			if(j!=ny-1)
				printf("%f,",field[i+j*nx]);
			else
				printf("%f",field[i+j*nx]);
		printf("\n");
	}
				
	//printf("Converged after %d iterations with maximum error %f\n",iter,error);
	free(field);
	free(nextfield);
}
