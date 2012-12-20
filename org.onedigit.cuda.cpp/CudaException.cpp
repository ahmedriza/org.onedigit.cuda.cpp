/*
 * CudaException.cpp
 *
 *  Created on: Dec 20, 2012
 *      Author: ahmed
 */

#include "CudaException.h"

CudaException::CudaException(const std::string& message) : message_(message)
{

}

CudaException::~CudaException() throw()
{

}

const char* CudaException::what() const throw()
{
	return message_.c_str();
}

