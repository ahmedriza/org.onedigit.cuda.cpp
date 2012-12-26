#include <iostream>
#include <stdio.h>
#include "CudaUtil.h"
#include "Matrix.h"
#include "CudaEventRecord.h"

void allocateMatrices(Matrix& A, Matrix& B, Matrix& C, int N);

#define BLOCK_SIZE 16

namespace {
	// solution:
	// 30 36 42 66 81 96 102 126 150
#ifdef CHECK_MATRIX
	void checkError(int i, const Matrix& matrix, int count, float expected)
	{
		float v = matrix.elements[i];
		float error = fabs(v - expected);
		if (error > 1.0e-6) {
			std::cout << "FAILURE at: " << count << std::endl;
			// std::cout << matrix << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	void checkElement(int i, const Matrix& matrix, int count)
	{
		switch (i) {
		case 0:
			checkError(i, matrix, count, 30.0);
			break;
		case 1:
			checkError(i, matrix, count, 36.0);
			break;
		case 2:
			checkError(i, matrix, count, 42.0);
			break;
		case 3:
			checkError(i, matrix, count, 66.0);
			break;
		case 4:
			checkError(i, matrix, count, 81.0);
			break;
		case 5:
			checkError(i, matrix, count, 96.0);
			break;
		case 6:
			checkError(i, matrix, count, 102.0);
			break;
		case 7:
			checkError(i, matrix, count, 126.0);
			break;
		case 8:
			checkError(i, matrix, count, 150.0);
			break;
		}

	}

	void printMatrix(const Matrix& matrix)
	{
		static int count = 0;
		for (int i = 0; i < matrix.height * matrix.width; i++) {
			checkElement(i, matrix, count);
			// std::cout << matrix.elements[i] << std::endl;
		}
		/*
		for (int i = 0; i < matrix.height; i++) {
			for (int j = 0; j < matrix.width; j++) {
				float v = matrix.elements[i * matrix.height + j];
				if (i == 2 && j == 0) {
					float error = fabs(v - 102.0);
					if (error > 1.0e-6) {
						std::cout << "FAILURE at: " << count << std::endl;
						std::cout << v << std::endl;
						exit(EXIT_FAILURE);
					}
				}
				// std::cout << v << std::endl;
			}
			// std::cout << std::endl;
		}
		*/
		count++;	
	}
#endif
}

// Get a mtrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
	// printf("GetElement, row = %d, col = %d\n", row, col);
	return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value)
{
	A.elements[row * A.stride + col] = value;
}

// locate col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	A.width = BLOCK_SIZE;
	A.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}

__global__ void MatMultSharedKernel(Matrix A, Matrix B, Matrix C)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	// Each thread computes 1 element of Csub accumulating results into Cvalue
	float Cvalue = 0.0;

	// Thread row and column within Csub
	int row = threadIdx.y, col = threadIdx.x;
	// Loop over all the sub-matrices of A and B required to compute Csub
	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		// Ge sub-matrices Asub of A and Bsub of B
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		Matrix Bsub = GetSubMatrix(B, m, blockCol);

		// Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);

		__syncthreads();

		// Multiply Asub and Bsub together
		for (int e = 0; e < BLOCK_SIZE; ++e) {
			Cvalue += As[row][e] * Bs[e][col];
		}

		__syncthreads();
	}

	// Each thread writes one element of Csub to memory
	SetElement(Csub, row, col, Cvalue);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// Each thread computes one element of C by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= A.height || col >= B.width) return;
	for (int e = 0; e < A.width; ++e) {
		Cvalue += ((A.elements[row * A.width + e]) * (B.elements[e * B.width + col]));
		/*
		printf("A(%d,%d) = %f, B(%d,%d) = %f, Cvalue = %f\n", 
			row, col, A.elements[row*A.width + e],
			row, col, B.elements[e*B.width + col], Cvalue);
		*/
	}
	C.elements[row * C.width + col] = Cvalue;
}

void testMatMul()
{
	// CudaUtil::getDeviceProperties(__LINE__, __FILE__);

	/*
	int height = 3, width = 3;
	int size = height * width * sizeof(float);

	Matrix A;
	A.height = height;
	A.width = width;
	A.elements = (float*)malloc(size);
	A.elements[0] = 1.0; A.elements[1] = 2.0; A.elements[2] = 3.0;
	A.elements[3] = 4.0; A.elements[4] = 5.0; A.elements[5] = 6.0;
	A.elements[6] = 7.0; A.elements[7] = 8.0; A.elements[8] = 9.0;

	Matrix B;
	B.height = height;
	B.width = width;
	B.elements = (float*)malloc(size);
	B.elements[0] = 1.0; B.elements[1] = 2.0; B.elements[2] = 3.0;
	B.elements[3] = 4.0; B.elements[4] = 5.0; B.elements[5] = 6.0;
	B.elements[6] = 7.0; B.elements[7] = 8.0; B.elements[8] = 9.0;

	Matrix C;
	C.height = height;
	C.width = width;
	C.elements = (float*)malloc(size);
	// Device matrices
	Matrix d_A;
	d_A.height = height;
	d_A.width = width;
	d_A.elements = (float*)malloc(size);

	Matrix d_B;
	d_B.height = height;
	d_B.width = width;
	d_B.elements = (float*)malloc(size);

	Matrix d_C;
	d_C.height = height;
	d_C.width = width;
	d_C.elements = (float*)malloc(size);
	*/

	int N = 4096;
	Matrix A, B, C;
	Matrix d_A, d_B, d_C;

	// int size = N * N * sizeof(float);

	allocateMatrices(A, B, C, N);

	d_A.width = N; d_A.height = N; d_A.stride = N;
	d_B.width = N; d_B.height = N; d_B.stride = N;
	d_C.width = N; d_C.height = N; d_C.stride = N;

	try {
		CudaEventRecord eventRecord;

		// CudaUtil::cudaCheckMalloc((void**)&d_A.elements, size, __LINE__, __FILE__);
		// CudaUtil::cudaCheckMalloc((void**)&d_B.elements, size, __LINE__, __FILE__);
		// CudaUtil::cudaCheckMalloc((void**)&d_C.elements, size, __LINE__, __FILE__);
		size_t pitch;
		CudaUtil::cudaCheckMallocPitch((void**)&d_A.elements, &pitch, N * sizeof(float), N, __LINE__, __FILE__);
		CudaUtil::cudaCheckMallocPitch((void**)&d_B.elements, &pitch, N * sizeof(float), N, __LINE__, __FILE__);
		CudaUtil::cudaCheckMallocPitch((void**)&d_C.elements, &pitch, N * sizeof(float), N, __LINE__, __FILE__);

		// copy to GPU
		// CudaUtil::cudaCheckMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice, __LINE__, __FILE__);
		// CudaUtil::cudaCheckMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice, __LINE__, __FILE__);
		CudaUtil::cudaCheckMemcpy2D(d_A.elements, pitch, A.elements, pitch, N * sizeof(float), N, cudaMemcpyHostToDevice, __LINE__, __FILE__);
		CudaUtil::cudaCheckMemcpy2D(d_B.elements, pitch, B.elements, pitch, N * sizeof(float), N, cudaMemcpyHostToDevice, __LINE__, __FILE__);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid( (B.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
		std::cout << "Preparing to run kernel..." << std::endl;

		int nIter = 10;
		for (int i = 0; i < nIter; i++) {
			// MatMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C);
			// Shared memory kernel is faster, almost twice as fast on Quadro 4000
			MatMultSharedKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C);
			CudaUtil::cudaCheckLastError(__LINE__, __FILE__);
		}
		cudaThreadSynchronize();
		CudaUtil::cudaCheckLastError(__LINE__, __FILE__);
		// copy to CPU
		// CudaUtil::cudaCheckMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost, __LINE__, __FILE__);
		CudaUtil::cudaCheckMemcpy2D(C.elements, pitch, d_C.elements, pitch, N * sizeof(float), N, cudaMemcpyDeviceToHost, __LINE__, __FILE__);

		cudaFree(d_A.elements);
		cudaFree(d_B.elements);
		cudaFree(d_C.elements);

	    // Compute and print the performance
		eventRecord.stop();
		float msecTotal = eventRecord.getTotalTime();
	    float msecPerMatrixMul = msecTotal / nIter;
	    std::cout << "Toatl time = " << msecTotal << " ms, time per matrix multiplication = " << msecPerMatrixMul << " ms" << std::endl;
	    double flopsPerMatrixMul = 2.0 * (double)N * (double)N * (double)N;
	    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	    std::cout << "Performance = " << gigaFlops << " GFlops/s" << std::endl;

	} catch (const std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		exit(EXIT_FAILURE);
	}

	// printMatrix(C);
	std::cout << A.elements[N*N-1] << std::endl;
	std::cout << B.elements[N*N-1] << std::endl;
	std::cout << C.elements[N*N-1] << std::endl;

	cudaFreeHost(A.elements);
	cudaFreeHost(B.elements);
	cudaFreeHost(C.elements);
}
