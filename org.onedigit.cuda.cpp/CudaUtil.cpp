/*
 * CudaUtil.cpp
 *
 *  Created on: Dec 20, 2012
 *      Author: ahmed
 */

#include <iostream>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include "CudaUtil.h"
#include "CudaException.h"

CudaUtil::CudaUtil()
{
}

CudaUtil::~CudaUtil()
{
}

void CudaUtil::cudaCheckFree(void* ptr, int line, const char* file)
{
	cudaError_t error = cudaFree(ptr);
	if (error != cudaSuccess) {
		std::ostringstream os;
		os << "cudaFree returned error code " << error << ", line " << line << ", in file " << file;
		throw CudaException(os.str());
	}
}

void CudaUtil::getDeviceProperties(int line, const char* file)
{
	int count;
	int error = cudaGetDeviceCount(&count);
	if (error != cudaSuccess) {
		std::ostringstream os;
		os << "cudaGetDeviceCount returned error code " << error << ", line " << line << ", in file " << file;
		throw CudaException(os.str());
	}
	std::cout << "Number of CUDA devices = " << count << std::endl;
	cudaDeviceProp prop;
	for (int i = 0; i < count; i++) {
		error = cudaGetDeviceProperties(&prop, i);
		if (error == cudaSuccess) {
			std::cout << "\tName: " << prop.name << std::endl;
			std::cout << "\tCompute capability: " << prop.major << "." << prop.minor << std::endl;
			// std::cout << "\tTotal Global Mem: " << prop.totalGlobalMem << std::endl;
		}
	}
}

cublasHandle_t CudaUtil::cublasInit()
{
	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		throw CudaException("CUBALS initialisation error");
	}
	return handle;
}

void CudaUtil::cublasClose(cublasHandle_t handle)
{
	cublasStatus_t status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		throw CudaException("CUBALS destroy error");
	}
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

void CudaUtil::cudaCheckMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height, int line, const char* file)
{
	int error = cudaMallocPitch(ptr, pitch, width, height);
    if (error != cudaSuccess) {
    	std::ostringstream os;
    	os << "cudaMallocPitch returned error code " << error << ", line " << line << ", in file " << file;
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

void CudaUtil::cudaCheckMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
		size_t width, size_t height, enum cudaMemcpyKind kind, int line, const char* file)
{
	int error = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
    if (error != cudaSuccess) {
    	std::ostringstream os;
    	os << "cudaMemcpy2D returned error code " << error << ", line " << line << ", in file " << file;
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

void CudaUtil::checkCublasStatus(cublasStatus_t status, int line, const char* file)
{
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream os;
		os << "CUBLAS error, line " << line << ", in file " << file;
		throw CudaException(os.str());
	}
}

void CudaUtil::cublasCheckSetVector(int n, int elemSize, void* hostPtr, int incx, void* devicePtr, int incy, int line, const char* file)
{
	cublasStatus_t status = cublasSetVector(n, elemSize, hostPtr, incx, devicePtr, incy);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream os;
		os << "CUBALS device write error, line " << line << ", in file " << file;
		throw CudaException(os.str());
	}
}

void CudaUtil::cublasCheckGetVector(int n, int elemSize, void* devicePtr, int incx, void* hostPtr, int incy, int line, const char* file)
{
	cublasStatus_t status = cublasGetVector(n, elemSize, devicePtr, incx, hostPtr, incy);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream os;
		os << "CUBALS device read error, line " << line << ", in file " << file;
		throw CudaException(os.str());
	}
}
