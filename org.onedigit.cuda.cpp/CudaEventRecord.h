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
	void stop();
	float getTotalTime() { return msecTotal_; }
private:
	 cudaEvent_t start_;
	 cudaEvent_t stop_;
	 float msecTotal_;
	 bool stopped_;
};

#endif /* CUDAEVENTRECORD_H_ */
