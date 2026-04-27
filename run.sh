#!/bin/bash
#SBATCH -c 20
#SBATCH -p igpuq
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1

module load nvhpc

nvc -fopenmp gpuA.c -O3 -mp=gpu -gpu=cc70,cc80 -o gpuA
nvc -fopenmp gpuB.c -O3 -mp=gpu -gpu=cc70,cc80 -o gpuB
nvc -fopenmp gpuC.c -O3 -mp=gpu -gpu=cc70,cc80 -o gpuC
nvcc -O3 cuda.cu -arch=compute_70 --gpu-code=sm_70,sm_80 -o cuda
gcc -fopenmp -lm -O3 openmpA.c -o openmpA
gcc -fopenmp -lm -O3 openmpB.c -o openmpB
gcc -fopenmp -lm -O3 openmpC.c -o openmpC
gcc -fopenmp -lm -O3 openmpD.c -o openmpD
gcc -lm -O3 singleA.c -o singleA
gcc -lm -O3 singleB.c -o singleB
gcc -lm -O3 singleC.c -o singleC
gcc -lm -O3 singleD.c -o singleD
nvfortran -fopenmp gpuFortran.f90 -O3 -mp=gpu -gpu=cc70,cc80 -o gpuFortran

echo "Running single threaded tests"
for EXE in singleA singleB singleC singleD
do
	time ./${EXE} 7000 7000 12345678 
done

echo "Running openmp tests"
for EXE in openmpA openmpB openmpC openmpD
do
	time ./${EXE} 7000 7000 12345678 
done

echo "Running gpu tests"
for EXE in gpuA gpuB gpuC cuda
do
	time ./${EXE} 7000 7000 12345678 
done


