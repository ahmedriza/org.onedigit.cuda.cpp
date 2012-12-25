#include <iostream>
#include <cublas_v2.h>
#include "CudaException.h"
#include "CudaUtil.h"
#include "CudaEventRecord.h"

#define N 7000

template <typename T>
void printMatrix(T* matrix)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << matrix[i * N + j] << " ";
		}
		std::cout << std::endl;
	}

}

template <typename T> struct CublasGemm;

template<>
struct CublasGemm<double>
{
	cublasStatus_t operator()(cublasHandle_t handle,
			cublasOperation_t transa,
			cublasOperation_t transb,
			int m,
			int n,
			int k,
			const double *alpha,
			const double *A,
			int lda,
			const double *B,
			int ldb,
			const double *beta,
			double *C,
			int ldc)
	{
		cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha, A, N, B, N, beta, C, N);
		return status;
	}
};

template<>
struct CublasGemm<float>
{
	cublasStatus_t operator()(cublasHandle_t handle,
			cublasOperation_t transa,
			cublasOperation_t transb,
			int m,
			int n,
			int k,
			const float *alpha,
			const float *A,
			int lda,
			const float *B,
			int ldb,
			const float *beta,
			float *C,
			int ldc)
	{
		cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha, A, N, B, N, beta, C, N);
		return status;
	}
};

void testCublas()
{
	typedef float Real;

	Real *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
	size_t size = N * N * sizeof(Real);
	try {
		CudaEventRecord eventRecord;
		cublasHandle_t handle = CudaUtil::cublasInit();
		std::cout << "CUBALS initialised" << std::endl;
		std::cout << "Allocating host memory" << std::endl;
		h_A = CudaUtil::hostCheckMalloc<Real>(size, __LINE__, __FILE__);
		h_B = CudaUtil::hostCheckMalloc<Real>(size, __LINE__, __FILE__);
		h_C = CudaUtil::hostCheckMalloc<Real>(size, __LINE__, __FILE__);
		std::cout << "Host memory allocated" << std::endl;
		// fill the matrices with test data
		for (size_t i = 0; i < N*N; i++) {
			h_A[i] = i; // rand() / (float)RAND_MAX;
			h_B[i] = i; // rand() / (float)RAND_MAX;
			h_C[i] = i; // rand() / (float)RAND_MAX;
		}
		std::cout << "Allocating device memory" << std::endl;
		d_A = CudaUtil::cudaCheckMalloc<Real>(size, __LINE__, __FILE__);
		d_B = CudaUtil::cudaCheckMalloc<Real>(size, __LINE__, __FILE__);
		d_C = CudaUtil::cudaCheckMalloc<Real>(size, __LINE__, __FILE__);
		std::cout << "Device memory allocated" << std::endl;
		std::cout << "Copying to device memory" << std::endl;
		CudaUtil::cublasCheckSetVector(N*N, sizeof(Real), h_A, 1, d_A, 1, __LINE__, __FILE__);
		CudaUtil::cublasCheckSetVector(N*N, sizeof(Real), h_B, 1, d_B, 1, __LINE__, __FILE__);
		CudaUtil::cublasCheckSetVector(N*N, sizeof(Real), h_C, 1, d_C, 1, __LINE__, __FILE__);
		std::cout << "Finished copying to device memory" << std::endl;
		Real alpha = 1.0;
		Real beta = 0.0;
		std::cout << "Calling CUBLAS GEMM" << std::endl;

		CublasGemm<Real> gemm;
		cublasStatus_t status = gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
		// cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
		// cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

		CudaUtil::checkCublasStatus(status, __LINE__, __FILE__);
		std::cout << "Finished calling CUBLAS GEMM" << std::endl;
		// cudaDeviceSynchronize();
		// Read the results back
		CudaUtil::cublasCheckGetVector(N*N, sizeof(Real), d_C, 1, h_C, 1, __LINE__, __FILE__);
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
