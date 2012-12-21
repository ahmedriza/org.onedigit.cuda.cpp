/*
 * Main.cpp
 *
 *  Created on: Dec 20, 2012
 *      Author: ahmed
 */

extern void testMatMul();

int main(int argc, char** argv)
{
	// vecAdd();
	// test();
	int N = 1000000;
	for (int i = 0; i < N; i++) {
		testMatMul();
	}
	return 0;
}


