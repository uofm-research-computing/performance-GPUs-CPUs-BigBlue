#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef PRECISION
	#define PRECISION double
#endif

struct neigh {
	PRECISION n, s, w, e, c;
};

#ifdef __CUDACC__
	__device__ __host__
#endif
inline int flattenCoord(const int i, const int j, const int nx, const int ny)
{
	return i+j*nx;
}

#ifdef __CUDACC__
	__device__ __host__
#endif
inline struct neigh getNeighbors(const int i, const int j, const int nx, const int ny, const PRECISION *field, const struct neigh boundary)
{
	/** returns elements from field with the following coordinates
	 *     n              i,j+1
	 *   w c e == i-1,j   i,j   i+1,j
	 *     s              i,j-1
	 **/
	struct neigh n;
	n.c=field[flattenCoord(i,j,nx,ny)];
	n.w=(i-1==-1)?boundary.w:field[flattenCoord(i-1,j,nx,ny)];
	n.e=(i+1==nx)?boundary.e:field[flattenCoord(i+1,j,nx,ny)];
	n.n=(j+1==ny)?boundary.n:field[flattenCoord(i,j+1,nx,ny)];
	n.s=(j-1==-1)?boundary.s:field[flattenCoord(i,j-1,nx,ny)];
	return n;
}

#ifdef __CUDACC__
	__device__ __host__
#endif
inline PRECISION fdm(const struct neigh n)
{
	/** Stencil for this pattern explicitly solving heat equation
	 *     n
	 *   w c e
	 *     s
	 **/
	return (n.n+n.s+n.e+n.w)*0.25;
}

