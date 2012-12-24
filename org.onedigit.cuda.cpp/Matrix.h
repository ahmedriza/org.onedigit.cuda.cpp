/*
 * Matrix.h
 *
 *  Created on: Dec 20, 2012
 *      Author: ahmed
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>

struct Matrix
{
	/*
	Matrix(int h, int w) : height(h), width(w)
	{
		int size = height * width * sizeof(float);
		elements = (float*)malloc(size);
		memset(elements, 0, size);
	}
	*/
	int height;
	int width;
	float* elements;
	int stride;
};

std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
{
	for (int i = 0; i < matrix.height * matrix.width; i++) {
		os << matrix.elements[i] << " ";
	}
	return os;
}

#endif /* MATRIX_H_ */
