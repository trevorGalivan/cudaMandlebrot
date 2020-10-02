#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "number_types.cuh"

// Detirmines the shade of an individual pixel based off of the number of iterations it took to escape, and its escape value
__device__
unsigned char shade(Cplx Zn, int iter, int maxIter, int blackDepth);