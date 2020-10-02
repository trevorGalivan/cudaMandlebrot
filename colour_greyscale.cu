#include <cmath>
#include "number_types.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// Detirmines the shade of an individual pixel based off of the number of iterations it took to escape, and its escape value
__device__
unsigned char shade(Cplx Zn, int iter, int maxIter, int blackDepth) {
	blackDepth++;

	if (iter == maxIter) {
		return 255;
	}

	double smoothDwell = double(iter) + 1. - log2(0.5 * log(norm(Zn)));
	smoothDwell = ((sqrt(50 * smoothDwell) - sqrt(50 * (double)blackDepth)) * (1 + 0.1 * logf(blackDepth)));
	
	if (smoothDwell > 180) { // Scales down higher values
		smoothDwell = sqrt(smoothDwell - 180) * 0.5 + 180;
	}

	return fmin(smoothDwell, 220.); // Clamps down very high values so that the actual mandlebrot set is always contrasted with surroundings
}