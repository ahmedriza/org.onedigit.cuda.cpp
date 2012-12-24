/*
 * Main.cpp
 *
 *  Created on: Dec 20, 2012
 *      Author: ahmed
 */

void testMatMul();
void testHostMatMult();

int main(int argc, char** argv)
{
	// vecAdd();
	// test();
	// int N = 1000000;
	int N = 1;
	for (int i = 0; i < N; i++) {
		testMatMul();
	}

	// testHostMatMult();

	return 0;
}


