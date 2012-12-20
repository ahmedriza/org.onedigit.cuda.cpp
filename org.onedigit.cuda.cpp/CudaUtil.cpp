/*
 * CudaUtil.cpp
 *
 *  Created on: Dec 20, 2012
 *      Author: ahmed
 */

#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaUtil.h"
#include "CudaException.h"

CudaUtil::CudaUtil()
{
	// TODO Auto-generated constructor stub

}

CudaUtil::~CudaUtil()
{
	// TODO Auto-generated destructor stub
}

void CudaUtil::cudaCheckMalloc(void** ptr, size_t size, int line, const char* file)
{
	int error = cudaMalloc(ptr, size);
    if (error != cudaSuccess) {
    	std::ostringstream os;
    	os << "cudaMalloc returned error code " << error << ", line " << line << ", in file " << file;
        throw CudaException(os.str());
    }
}

void CudaUtil::cudaCheckMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, int line, const char* file)
{
	int error = cudaMemcpy(dst, src, count, kind);
    if (error != cudaSuccess) {
    	std::ostringstream os;
    	os << "cudaMemcpy returned error code " << error << ", line " << line << ", in file " << file;
    	throw CudaException(os.str());
    }
}

void CudaUtil::cudaCheckLastError(int line, const char* file)
{
	cudaError_t lauchSuccess = cudaGetLastError();
	if (lauchSuccess != cudaSuccess) {
		std::ostringstream os;
		os << "CUDA error " << cudaGetErrorString(lauchSuccess) << ", line " << line << ", in file " << file;
		throw CudaException(os.str());
	}
}
