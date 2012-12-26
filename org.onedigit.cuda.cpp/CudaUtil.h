/*
 * CudaUtil.h
 *
 *  Created on: Dec 20, 2012
 *      Author: ahmed
 */

#ifndef CUDAUTIL_H_
#define CUDAUTIL_H_

#include <stdlib.h>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include "CudaException.h"

class CudaUtil {
public:
	CudaUtil();
	virtual ~CudaUtil();
	static void cudaCheckFree(void* ptr, int line, const char* file);
	static void getDeviceProperties(int line, const char* file);
	static cublasHandle_t cublasInit();
	static void cublasClose(cublasHandle_t handle);

	template <typename T>
	static T* hostCheckMalloc(size_t size, int line, const char* file)
	{
		T* ptr = (T*)malloc(size);
		if (ptr == 0) {
			std::ostringstream os;
			os << "malloc failed, line " << line << ", in file " << file;
			throw CudaException(os.str());
		}
		return ptr;
	}

	template <typename T>
	static T* cudaCheckMalloc(size_t size, int line, const char* file)
	{
		T* ptr = 0;
		int error = cudaMalloc(&ptr, size);
	    if (error != cudaSuccess) {
	    	std::ostringstream os;
	    	os << "cudaMalloc returned error code " << error << ", line " << line << ", in file " << file;
	    	throw CudaException(os.str());
	    }
	    return (T*)ptr;
	}

	static void cudaCheckMalloc(void** ptr, size_t size, int line, const char* file);
	static void cudaCheckMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height, int line, const char* file);

	static void cudaCheckMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, int line, const char* file);
	static void cudaCheckMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
			size_t width, size_t height, enum cudaMemcpyKind kind, int line, const char* file);

	static void cudaCheckLastError(int line, const char* file);

	static void checkCublasStatus(cublasStatus_t status, int line, const char* file);
	static void cublasCheckSetVector(int n, int elemSize, void* hostPtr, int incx, void* devicePtr, int incy, int line, const char* file);
	static void cublasCheckGetVector(int n, int elemSize, void* devicPtr, int incx, void* hostPtr, int incy, int line, const char* file);
};

#endif /* CUDAUTIL_H_ */
