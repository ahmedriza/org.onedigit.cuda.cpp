
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N (8388480 * 2)

struct Matrix
{
	Matrix(int h, int w) : height(h), width(w) 
	{
		int size = height * width * sizeof(float);
		elements = (float*)malloc(size);
		memset(elements, 0, size);
	}
	int height;
	int width;
	float* elements;
};

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

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) 
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > A.height || col > B.width) return;
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
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

void testMatMul()
{
	int height = 3;
	int width = 3;
	Matrix A(height, width);
	A.elements[0] = 1.0;
	A.elements[1] = 2.0;
	A.elements[2] = 3.0;
	A.elements[3] = 4.0;
	A.elements[4] = 5.0;
	A.elements[5] = 6.0;
	A.elements[6] = 7.0;
	A.elements[7] = 8.0;
	A.elements[8] = 9.0;

	Matrix B(height, width);
	B.elements[0] = 1.0;
	B.elements[1] = 2.0;
	B.elements[2] = 3.0;
	B.elements[3] = 4.0;
	B.elements[4] = 5.0;
	B.elements[5] = 6.0;
	B.elements[6] = 7.0;
	B.elements[7] = 8.0;
	B.elements[8] = 9.0;

	Matrix C(height, width);

	// Device matrices

	Matrix d_A(height, width), d_B(height, width), d_C(height, width);

	int size = height * width * sizeof(float);
	
	int error = cudaMalloc( (void**)&d_A.elements, size );
    if (error != cudaSuccess) {
        printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	error = cudaMalloc( (void**)&d_B.elements, size );
    if (error != cudaSuccess) {
        printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	error = cudaMalloc( (void**)&d_C.elements, size );
    if (error != cudaSuccess) {
        printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	// copy to GPU

	error = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("cudaMemcpy returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	error = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("cudaMemcpy returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	dim3 dimBlock(16, 16);
	dim3 dimGrid( (B.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	MatMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C);

	cudaError_t lauchSuccess = cudaGetLastError();
	if (lauchSuccess != cudaSuccess) {
        printf("Lauching vecAddDevice returned error %s, line(%d)\n", cudaGetErrorString(lauchSuccess), __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy to CPU

	error =	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("cudaMemcpy returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	
	for (int i = 0; i < height * width; i++) {
		std::cout << C.elements[i] << std::endl;
	}
}

int main()
{
	// vecAdd();
	// test();	
	testMatMul();
	return 0;
}

