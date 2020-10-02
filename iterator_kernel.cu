#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "colour_greyscale.cuh"
#include "number_types.cuh"

constexpr double escapeDistSquared = 20.*20.;

/*  Kernel to create an image of a region of the mandelbrot set, defines by bounds.
 *  Parameters:
 *    res - defines the final resolution of the output image. 
 *    bounds - defines the region of the mandlebrot set to be rendered. Treated 'x' and 'y' represent the lower and upper bounds for the real component
 *             'z' and 'w' represent the lower and upper bounds for the imaginary component
 *    blackDepth - represents the expected minimum for the number of iterations it will take for any pixel to escape. Used to progressibly refine the
 *                 colour scheme so that we always use our full shading range from black to white
 *    maxIter - maximum iteration cap for this round
 *    output - pointer is used to output each pixel as a signel unsigned character. MUST be a block of device memory at least res.x * res.y bytes long
 *    nextBlackDepth - Used to record the lowest number of iterations any pixel took to escape. Set the value pointed to by this pointer to a very high
 *                     value before launching the kernel. Pointer should point to a single int on device memory
 *    nextMaxIter - similar to nextBlackDepth, but samples one thread in each block, if this thread did not take the maximum number of iterations, then
 *                  it's iteration count is checked against the current value, and updated if larger. Pointer should point to single int on device memory
**/
__global__
void mandleBrot(uint2 res, double4 bounds, int blackDepth, int maxIter, unsigned char* output, int* nextBlackDepth, int* nextMaxIter)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < res.x * res.y; i += stride) { // Samples all pixels in the range specified, using coalesced memory access
        Cplx C;
        C.re = bounds.x + (bounds.y - bounds.x) / (double)res.x * (double)(i % res.x);
        C.im = bounds.z + (bounds.w - bounds.z) / (double)res.y * (double)(i / res.x);
        Cplx Z = C;

        // Iterates until our number escapes
        int iter = 0;
        while (norm(Z) < escapeDistSquared && iter < maxIter) {
            iter++;
            Z = square(Z) + C;
        }

        // Set nextBlackDepth to the min of every threads iter without escape conditions
        atomicMin(nextBlackDepth, iter);

        if (iter != maxIter && threadIdx.x == 0) {
            atomicMax(nextMaxIter, iter);
        }

        output[i] = shade(Z, iter, maxIter, blackDepth); 
    }
}