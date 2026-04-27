# Comparing the performance of using GPUs vs CPUs
This is a set of examples for using GPUs on the BigBlue using a simple Finite Difference Method (FDM) solution of the heat equation. Primarily this is just to show the simplest parallelization strategies, transform and reduce, to decrease runtime. The examples are compiled using the nvhpc module on the cluster. Most of the examples are in C, but a Fortran version is also included. The point is to show off a cross platform library, OpenMP, that can be used on other GPUs. For example, you could use the AMD HIP or Intel OneAPI libraries to compile any of the OpenMP codes and rerun the tests. The only exception is the cuda.c version that can only run on NVIDIA GPUs. The results might vary depending on the dimensions, seed, and GPU. For this example I use a 7000 by 7000 grid with a 1000 iteration depth on the igpuq partition of BigBlue using 20 cores on an NVIDIA V100 GPU so the job completes quickly.

## Running the tests
On BigBlue, you can simply download this repo and submit by running:
```
git clone https://github.com/uofm-research-computing/performance-GPUs-CPUs-BigBlue.git
cd performance-GPUs-CPUs-BigBlue
sbatch run.sh
```

### Single core tests
All singleN.c sources using a single processor with the following properties:
1. A. This is the basic double precision version of the algorithm in the presentation.
2. B: This is the basic single precision version of the algorithm in the presentation.
3. C: This is a modified version of the algorithm in the presentation. This one splits the error update term.
4. D: This is a modified version of the algorithm in the presentation. This one merges the update loop.

### Multicore tests
All openmpN.c sources using a single processor with the following properties:
1. A. This is the basic double precision version of the algorithm in the presentation.
2. B: This is the basic single precision version of the algorithm in the presentation.
3. C: This is a modified version of the algorithm in the presentation. This one splits the error update term.
4. D: This is a modified version of the algorithm in the presentation. This one merges the update loop.

### GPU tests
All gpuN.c sources using a single processor and GPU with the following properties:
1. A. This is a modified version of the double precision algorithm in the presentation. This one splits the error update term and combines it with swap.
2. B. This is a modified version of the single precision algorithm in the presentation. This one splits the error update term and combines it with swap.
3. C: This is a modified version of the single precision algorithm in the presentation. This one splits the error update term and combines it with swap. This one splits the update loop into a nested loop. There is the possibility of combining this with the "collapse" clause in openmp to recreate the B test.
4. Fortran: This is a modified version of the algorithm in the presentation. This one splits the error update term and combines it with swap. This one merges the update loop.
5. A cuda.cu version is included as a reference. This one swaps the device pointers to make the swap faster. This one is pretty slow for such a low level version.

## Graph used in presentation
Using a version of openmpB.c, createGraph.c will output the csv and heatmap.py will write a heatmap.png file. You can just run:
```
./makeGraph.sh
```
This one will take some time since the convergence tolerance is much smaller, but it produces a nice saddle shaped graph.

## Citations and other reading material
[NVHPC openmp from NVIDIA sdk guide](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/index.html#openmp-use)
[a much better description of the FDM algorithm using a much better version of the CUDA code](https://enccs.github.io/OpenACC-CUDA-beginners/2.02_cuda-heat-equation/)
[AMD HIP examples](https://github.com/amd/HPCTrainingExamples/tree/main/HIP-OpenMP)
[Intel OneAPI OpenMP offloading tuning guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/openmp-offloading-tuning-guide.html)

