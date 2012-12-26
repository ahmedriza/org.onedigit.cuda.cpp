/*
 * CudaEventRecord.cpp
 *
 *  Created on: Dec 25, 2012
 *      Author: ahmed
 */

#include <iostream>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaEventRecord.h"
#include "CudaException.h"

CudaEventRecord::CudaEventRecord() : msecTotal_(0.0), stopped_(false)
{
	cudaError_t error = cudaEventCreate(&start_);
    if (error != cudaSuccess)
    {
    	std::ostringstream os;
    	os << "Failed to create start event, error code: " << cudaGetErrorString(error);
    	throw CudaException(os.str());
    }

    error = cudaEventCreate(&stop_);
    if (error != cudaSuccess)
    {
    	std::ostringstream os;
    	os << "Failed to create stop event, error code: " << cudaGetErrorString(error);
    	throw CudaException(os.str());
    }

    error = cudaEventRecord(start_, NULL);
    if (error != cudaSuccess)
    {
    	std::ostringstream os;
    	os << "Failed to record start event, error code: " << cudaGetErrorString(error);
    	throw CudaException(os.str());
    }
}

CudaEventRecord::~CudaEventRecord()
{
	if (!stopped_) {
		stop();
	}
}

void CudaEventRecord::stop()
{
	if (!stopped_) {
	    // Record the stop event
		cudaError_t error = cudaEventRecord(stop_, NULL);
		if (error != cudaSuccess)
		    {
		    	std::ostringstream os;
		    	os << "Failed to record stop event, error code: " << cudaGetErrorString(error);
		    	throw CudaException(os.str());
		    }

	    // Wait for the stop event to complete
	    error = cudaEventSynchronize(stop_);
	    if (error != cudaSuccess) {
	    	std::ostringstream os;
	    	os << "Failed to synchronize on the stop event, error code: " << cudaGetErrorString(error);
	    	throw CudaException(os.str());
	    }

	    msecTotal_ = 0.0f;
	    error = cudaEventElapsedTime(&msecTotal_, start_, stop_);
	    if (error != cudaSuccess) {
	    	std::ostringstream os;
	    	os << "Failed to get time elapsed between events, error code: " << cudaGetErrorString(error);
	    	throw CudaException(os.str());
	    }
	    stopped_ = true;
	}
}

