//Simple cuda parallel single precision Jacobi iteration solver for a 2 dimensional heat equation
//This also updates the field by simply swapping pointers.
//Note: There is a limit to the number of elements in the kernel calls of 2^31-1
#define PRECISION float
#include "common.h"
#include "limits.h"

#define THRESHOLD 0.001
#define MAX_ITER 1000
#define THREADS_PER_BLOCK 256

//Update nextfield using field and obtain error per element
__global__
void step(PRECISION *field, PRECISION *nextfield, PRECISION *error, int nx, int ny, struct neigh boundary)
{
	uint offset=gridDim.x*blockDim.x;
	for(uint current=blockIdx.x* blockDim.x + threadIdx.x;current<nx*ny;current+=offset)
	{
		int i=current%nx;
		int j=(current/nx)%ny;
		struct neigh n=getNeighbors(i,j,nx,ny,field,boundary);
		nextfield[current]=fdm(n);
		error[current]=fabsf(nextfield[current]-field[current]);
	}
}

//Kernel for cuda_max reduction below
__global__ void max_reduce_kernel(PRECISION* f, int i, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx+i<N)
		f[idx]=fmaxf(f[idx],f[idx+i]);
}

//In place reduction, this is destructive to "in"
PRECISION cuda_max(PRECISION* in, int N)
{
	for(int i=N/2;i>0;i/=2)
	{
		int numThreads=min(THREADS_PER_BLOCK,i);
		int numBlocks=max(i/THREADS_PER_BLOCK,1);
		max_reduce_kernel<<<numBlocks,numThreads>>>(in,i,N);
	}
	cudaDeviceSynchronize();
	PRECISION result;
	cudaMemcpy(&result, in, sizeof(PRECISION), cudaMemcpyDeviceToHost);
	
	return result;
}

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
	PRECISION *field_d=NULL;
	PRECISION *nextfield_d=NULL;
	PRECISION *error_d=NULL;
	cudaMalloc((void **)&field_d, nx*ny*sizeof(PRECISION));
	cudaMalloc((void **)&nextfield_d, nx*ny*sizeof(PRECISION));
	cudaMalloc((void **)&error_d, nx*ny*sizeof(PRECISION));
	
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
	int N=nx*ny;
	cudaMemcpy((char *) field_d, (char*)field,nx*ny*sizeof(PRECISION), cudaMemcpyHostToDevice);
	do
	{
		int numThreads=min(THREADS_PER_BLOCK,N);
		int numBlocks=max(N/THREADS_PER_BLOCK,1);
		step<<<numBlocks,numThreads>>>(field_d, nextfield_d, error_d, nx, ny, boundary);
		cudaDeviceSynchronize();
		error=cuda_max(error_d,N);
		
		PRECISION *buf=field_d;
		field_d=nextfield_d;
		nextfield_d=buf;
		iter++;
	} while(error>THRESHOLD && iter<MAX_ITER);
	
	cudaMemcpy(field, field_d, sizeof(PRECISION)*N, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Converged after %d iterations with maximum error %f\n",iter,error);
	free(field);
	free(nextfield);
	cudaFree(field_d);
	cudaFree(nextfield_d);
}
