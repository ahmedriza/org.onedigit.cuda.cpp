#include <iostream>
#include "CudaUtil.h"
#include "Matrix.h"

namespace {
	void printMatrix(const Matrix& matrix)
	{
		static int count = 0;
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
		count++;	
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

void testMatMul()
{
	// CudaUtil::getDeviceProperties(__LINE__, __FILE__);

	int height = 3, width = 3;
	int size = height * width * sizeof(float);

	Matrix A(height, width);
	A.elements[0] = 1.0; A.elements[1] = 2.0; A.elements[2] = 3.0;
	A.elements[3] = 4.0; A.elements[4] = 5.0; A.elements[5] = 6.0;
	A.elements[6] = 7.0; A.elements[7] = 8.0; A.elements[8] = 9.0;

	Matrix B(height, width);
	B.elements[0] = 1.0; B.elements[1] = 2.0; B.elements[2] = 3.0;
	B.elements[3] = 4.0; B.elements[4] = 5.0; B.elements[5] = 6.0;
	B.elements[6] = 7.0; B.elements[7] = 8.0; B.elements[8] = 9.0;

	Matrix C(height, width);
	// Device matrices
	Matrix d_A(height, width), d_B(height, width), d_C(height, width);

	try {
		CudaUtil::cudaCheckMalloc((void**)&d_A.elements, size, __LINE__, __FILE__);
		CudaUtil::cudaCheckMalloc((void**)&d_B.elements, size, __LINE__, __FILE__);
		CudaUtil::cudaCheckMalloc((void**)&d_C.elements, size, __LINE__, __FILE__);
		// copy to GPU
		CudaUtil::cudaCheckMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice, __LINE__, __FILE__);
		CudaUtil::cudaCheckMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice, __LINE__, __FILE__);
		dim3 dimBlock(16, 16);
		dim3 dimGrid( (B.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
		MatMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C);
		CudaUtil::cudaCheckLastError(__LINE__, __FILE__);
		// copy to CPU
		CudaUtil::cudaCheckMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost, __LINE__, __FILE__);
	} catch (const std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		exit(EXIT_FAILURE);
	}

	printMatrix(C);
}
