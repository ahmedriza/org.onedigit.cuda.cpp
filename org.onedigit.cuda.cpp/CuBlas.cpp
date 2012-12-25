#include <iostream>
#include <cublas_v2.h>
#include "CudaException.h"
#include "CudaUtil.h"
#include "CudaEventRecord.h"

#define N 10000

void printMatrix(float* matrix)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << matrix[i * N + j] << " ";
		}
		std::cout << std::endl;
	}

}

void testCublas()
{
	try {
		CudaEventRecord eventRecord;

		cublasHandle_t handle = CudaUtil::cublasInit();

		size_t size = N * N * sizeof(float);
		float* h_A = CudaUtil::hostCheckMalloc<float>(size, __LINE__, __FILE__);
		float* h_B = CudaUtil::hostCheckMalloc<float>(size, __LINE__, __FILE__);
		float* h_C = CudaUtil::hostCheckMalloc<float>(size, __LINE__, __FILE__);

		// fill the matrices with test data
		for (size_t i = 0; i < N*N; i++) {
			h_A[i] = i; // rand() / (float)RAND_MAX;
			h_B[i] = i; // rand() / (float)RAND_MAX;
			// h_C[i] = i; // rand() / (float)RAND_MAX;
		}

		float* d_A = CudaUtil::cudaCheckMalloc<float>(size, __LINE__, __FILE__);
		float* d_B = CudaUtil::cudaCheckMalloc<float>(size, __LINE__, __FILE__);
		float* d_C = CudaUtil::cudaCheckMalloc<float>(size, __LINE__, __FILE__);

		CudaUtil::cublasCheckSetVector(N*N, sizeof(float), h_A, 1, d_A, 1, __LINE__, __FILE__);
		CudaUtil::cublasCheckSetVector(N*N, sizeof(float), h_B, 1, d_B, 1, __LINE__, __FILE__);
		// CudaUtil::cublasCheckSetVector(N*N, sizeof(float), h_C, 1, d_C, 1, __LINE__, __FILE__);

		float alpha = 1.0f;
		float beta = 0.0f;
		cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
		CudaUtil::checkCublasStatus(status, __LINE__, __FILE__);

		// Read the results back
		CudaUtil::cublasCheckGetVector(N*N, sizeof(float), d_C, 1, h_C, 1, __LINE__, __FILE__);

		// printMatrix(h_C);

		// memory cleanup
		free(h_A);
		free(h_B);
		free(h_C);
		CudaUtil::cudaCheckFree(d_A, __LINE__, __FILE__);
		CudaUtil::cudaCheckFree(d_B, __LINE__, __FILE__);
		CudaUtil::cudaCheckFree(d_C, __LINE__, __FILE__);

		CudaUtil::cublasClose(handle);

	} catch (std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		exit(EXIT_FAILURE);
	}
}
