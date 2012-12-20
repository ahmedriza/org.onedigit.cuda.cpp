
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaUtil.h"

#define N (8388480 * 2)

__global__ void add(int a, int b, int *c)
{
	*c = a + b;
}

__global__ void vecAddDevice(int* a, int* b, int* c)
{
	// int gx = gridDim.x;
	// int gy = gridDim.y;
	// int tid = blockIdx.x; 
	int tid = threadIdx.x + blockIdx.x * blockDim.x; 
	/* printf("tid = %d\n", tid); */
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		// Increment tid by the total number of threads running in the grid.
		// This is simply the number of threads per block * the number of blocks in the grid
		tid += blockDim.x * gridDim.x;
	}
}

void vecAdd()
{
	int* a = new int[N];
	int* b = new int[N];
	int* c = new int[N];
	int *dev_a, *dev_b, *dev_c;
	int error = cudaMalloc( (void**)&dev_a, N * sizeof(int) );
    if (error != cudaSuccess) {
        printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMalloc( (void**)&dev_b, N * sizeof(int) );
    if (error != cudaSuccess) {
        printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMalloc( (void**)&dev_c, N * sizeof(int) );
    if (error != cudaSuccess) {
        printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}
	error = cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("cudaMemcpy returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("cudaMemcpy returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	int numBlocks = 128;
	int threadsPerBlock = 128;
	vecAddDevice<<< numBlocks, threadsPerBlock >>>(dev_a, dev_b, dev_c);
	cudaError_t lauchSuccess = cudaGetLastError();
	if (lauchSuccess != cudaSuccess) {
        printf("Lauching vecAddDevice returned error %s, line(%d)\n", cudaGetErrorString(lauchSuccess), __LINE__);
		exit(EXIT_FAILURE);
	}
	
	error = cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("cudaMemcpy returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	/*
	for (int i = 0; i < N; i++) {
		std::cout << a[i] << "+" << b[i] << " = " << c[i] << std::endl;
	}
	*/
	delete[] a;
	delete[] b;
	delete[] c;
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

void test()
{
	int c;
	int* dev_c;
	int error = cudaMalloc( (void**)&dev_c, sizeof(int) );
    if (error != cudaSuccess) {
        printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	
	add<<<1, 1>>>(2, 7, dev_c);
	
	error =	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("cudaMemcpy returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	std::cout << "2 + 7 = " << c << std::endl;
	cudaFree(dev_c);
}


