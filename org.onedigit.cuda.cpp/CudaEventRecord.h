/*
 * CudaEventRecord.h
 *
 *  Created on: Dec 25, 2012
 *      Author: ahmed
 */

#ifndef CUDAEVENTRECORD_H_
#define CUDAEVENTRECORD_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CudaEventRecord {
public:
	CudaEventRecord();
	virtual ~CudaEventRecord();
private:
	 cudaEvent_t start_;
	 cudaEvent_t stop_;
};

#endif /* CUDAEVENTRECORD_H_ */
