#include <stdlib.h>
#include <time.h>
#include "Matrix.h"

std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
{
	for (int i = 0; i < matrix.height * matrix.width; i++) {
		os << matrix.elements[i] << " ";
	}
	return os;
}

// Mutliply two square matrices
/*
 * | a b c | | j k l | = | aj+bm+cp  ak+bn+cq  al+bo+cr |
 * | d e f | | m n o |   | dj+em+fp  dk+en+fq  dl+eo+fr |
 * | g h i | | p q r |   | gj+hm+ip  gk+hn+iq  gl+ho+ir |
 *
 * {{30, 36, 42}, {66, 81, 96}, {102, 126, 150}}
 *
 * 1 2 3 | 1 2 3 | = | 1*1+2*4+3*7  (30) 1*2+2*5+3*8  (36) 1*3+2*6+3*9  (42)
 * 4 5 6 | 4 5 6 |   | 4*1+5*4+6*7  (66) 4*2+5*5+6*8  (81) 4*3+5*6+6*9  (96)
 * 7 8 9 | 7 8 9 |   | 7*1+8*4+9*7 (102) 7*2+8*5+9*8 (126) 7*3+8*6+9*9 (150)
 */
void matMul(const Matrix& A, const Matrix& B, Matrix& C, int N)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float Cval = 0.0;
			for (int k = 0; k < N; k++) {
				int aindex = i * N + k;
				int bindex = k * N + j;
				float a = A.elements[i * N + k];
				float b = B.elements[k * N + j];
				// std::cout << "A[ " << aindex << "] = " << a << ", B[ " << bindex << "] = " << b << std::endl;
				Cval += a * b;
			}
			// std::cout << "*** Cval = " << Cval << std::endl;
			C.elements[i * N + j] = Cval;
		}
	}
}

void printMatrix(const Matrix& matrix)
{
	for (int i = 0; i < matrix.width; i++) {
		for (int j = 0; j < matrix.height; j++) {
			std::cout << matrix.elements[i * matrix.width + j] << " ";
		}
		std::cout << std::endl;
	}
}

void allocateMatrices(Matrix& A, Matrix& B, Matrix& C, int N)
{
	int size = N * N * sizeof(float);
	A.width = N; A.height = N; A.elements = (float*)malloc(size);
	B.width = N; B.height = N; B.elements = (float*)malloc(size);
	C.width = N; C.height = N; C.elements = (float*)malloc(size);

	/*
	A.elements[0] = 1; A.elements[1] = 2; A.elements[2] = 3;
	A.elements[3] = 4; A.elements[4] = 5; A.elements[5] = 6;
	A.elements[6] = 7; A.elements[7] = 8; A.elements[8] = 9;

	B.elements[0] = 1; B.elements[1] = 2; B.elements[2] = 3;
	B.elements[3] = 4; B.elements[4] = 5; B.elements[5] = 6;
	B.elements[6] = 7; B.elements[7] = 8; B.elements[8] = 9;
	*/
	for (int i = 0; i < N * N; i++) {
		A.elements[i] = i;
		B.elements[i] = i;
	}

}

void testHostMatMult()
{
	int N = 10000;
	Matrix A, B, C;
	allocateMatrices(A, B, C, N);
	time_t start = time(0);
	matMul(A, B, C, N);
	time_t end = time(0);
	double elapsed = difftime(end, start);
	std::cout << "Elapsed = " << elapsed << std::endl;

	// printMatrix(C);
	std::cout << A.elements[N*N-1] << std::endl;
	std::cout << B.elements[N*N-1] << std::endl;
	std::cout << C.elements[N*N-1] << std::endl;
}
