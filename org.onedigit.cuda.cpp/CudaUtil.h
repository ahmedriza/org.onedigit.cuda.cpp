/*
 * CudaUtil.h
 *
 *  Created on: Dec 20, 2012
 *      Author: ahmed
 */

#ifndef CUDAUTIL_H_
#define CUDAUTIL_H_

class CudaUtil {
public:
	CudaUtil();
	virtual ~CudaUtil();
	static void getDeviceProperties(int line, const char* file);
	static void cudaCheckMalloc(void** ptr, size_t size, int line, const char* file);
	static void cudaCheckMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, int line, const char* file);
	static void cudaCheckLastError(int line, const char* file);
};

#endif /* CUDAUTIL_H_ */
