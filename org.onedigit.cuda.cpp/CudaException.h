/*
 * CudaException.h
 *
 *  Created on: Dec 20, 2012
 *      Author: ahmed
 */

#ifndef CUDAEXCEPTION_H_
#define CUDAEXCEPTION_H_

#include <string>
#include <exception>

class CudaException : public std::exception {
public:
	CudaException(const std::string& message);
	virtual ~CudaException() throw();
	virtual const char* what() const throw();
private:
	const std::string message_;
};

#endif /* CUDAEXCEPTION_H_ */
