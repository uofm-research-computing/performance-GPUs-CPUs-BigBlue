//Simple gpu parallel single precision Jacobi iteration solver for a 2 dimensional heat equation
//This also updates the field by copying values from newfield to field.
#define PRECISION float
#include "common.h"
#include <omp.h>

#define THRESHOLD 0.001
#define MAX_ITER 1000

int main(int argc, char **argv)
{
	//Check that we have enough options
	if(argc!=4)
	{
		printf("Usage: %s nx ny seed\n",argv[0]);
		int num_devices = omp_get_num_devices();
		printf("Number of available devices %d\n", num_devices);
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

	//Step until error is almost zero
	PRECISION error;
	int iter=0;
	int N=nx*ny;
	#pragma omp target enter data map(to: field[0:N] , nextfield[0:N], boundary, N)
	do
	{
		#pragma omp target teams distribute parallel for simd
		for(int current=0;current<N;current++)
		{
			int i=current%nx;
			int j=(current/nx)%ny;
			struct neigh n=getNeighbors(i,j,nx,ny,field,boundary);
			nextfield[current]=fdm(n);
		}
		
		error=0;
		#pragma omp target map(tofrom:error)
		#pragma omp teams distribute parallel for simd reduction(max:error)
		for(int current=0;current<N;current++)
		{
			PRECISION derror=fabs(nextfield[current]-field[current]);
			error=fmaxf(error,derror);
			field[current]=nextfield[current];
		}
		iter++;
	} while(error>THRESHOLD && iter<MAX_ITER);
	#pragma omp target exit data map(from: field[0:N] , nextfield[0:N])
	printf("Converged after %d iterations with maximum error %f\n",iter,error);
	free(field);
	free(nextfield);
}
