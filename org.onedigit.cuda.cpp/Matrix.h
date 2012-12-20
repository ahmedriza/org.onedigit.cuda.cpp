/*
 * Matrix.h
 *
 *  Created on: Dec 20, 2012
 *      Author: ahmed
 */

#ifndef MATRIX_H_
#define MATRIX_H_

struct Matrix
{
	Matrix(int h, int w) : height(h), width(w)
	{
		int size = height * width * sizeof(float);
		elements = (float*)malloc(size);
		memset(elements, 0, size);
	}
	int height;
	int width;
	float* elements;
};

#endif /* MATRIX_H_ */
