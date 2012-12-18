
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define N 10

__global__ void add(int a, int b, int *c)
{
	*c = a + b;
}

__global__ void vecAddDevice(int* a, int* b, int* c)
{
	int gx = gridDim.x;
	int gy = gridDim.y;
	int tid = blockIdx.x; 
	/* printf("tid = %d\n", tid); */
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}

void getDevice()
{
	int count;
	int error =	cudaGetDeviceCount(&count);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    std::cout << "Number of devices = " << count << std::endl;
    cudaDeviceProp prop;
    for (int i = 0; i < count; i++) {
		error = cudaGetDeviceProperties(&prop, i);
		if (error == cudaSuccess) {
			std::cout << "Name: " << prop.name << std::endl;
			std::cout << "Total Global Mem: " << prop.totalGlobalMem << std::endl;
		}
	}
}

void vecAdd()
{
	getDevice();

	int a[N], b[N], c[N];
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

	vecAddDevice<<<N, 1>>>(dev_a, dev_b, dev_c);
	
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
}

void test()
{
	getDevice();

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

int main()
{
	vecAdd();
	// test();
	return 0;
}

