#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>

#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "device_launch_parameters.h"


#include "iterator_kernel.cuh"
#include "image_output.h"

#include "text_display.cuh"


using namespace std;



int main(int argc, char** argv)
{
    uint2 res = { 1028u * 4, 1028u * 4 };


    unsigned char* M;
    int* blackDepth;
    int* nextMaxIter;
    int tempBlackDepth;
    cudaMallocManaged(&blackDepth, sizeof(int));
    cudaMallocManaged(&nextMaxIter, sizeof(int));
    cudaMallocManaged(&M, res.x * res.y * sizeof(unsigned char));

    *nextMaxIter = 255; // Sets the initial iteration cap to 255

    // Run kernel on whole image
    //dim3 blockSize(32, 8);
    //dim3 numBlocks((xRes + blockSize.x - 1) / blockSize.x, (yRes + blockSize.y - 1) / blockSize.y);
    int blockSize = 256;
    int numBlocks = 256;

    double2 zoomCenter = { -0.7999075073, -0.16746163399 };
    double2 zoomBounds = { 1.5, 0.000002 }; // vertical radius at start of zoom, vertical radius at end of zoom
    double aspectRatio = 1.; // horizontal radius / vertical radius
    double radius = zoomBounds.x;
    int animLength = 400;
    double shrinkFactor = pow(zoomBounds.y / zoomBounds.x, 1. / animLength); // Factor by which the vertical radius of the image decreases each frame

    for (int i = 0; i < animLength; i++) {
        double4 bounds = { zoomCenter.x - radius* aspectRatio, zoomCenter.x + radius* aspectRatio,
                           zoomCenter.y - radius,              zoomCenter.y + radius              }; // xmin, xmax, ymin, ymax

        radius *= shrinkFactor;
        printf("Frame number %d , Max iteration cap:%d  ", i, *nextMaxIter);

        tempBlackDepth = *blackDepth;
        *blackDepth = 10000000;
        std::cout << shrinkFactor;
        // Launches the kernel
        mandleBrot << <numBlocks, blockSize >> > (res, bounds, tempBlackDepth, *nextMaxIter * 1.05, M, blackDepth, nextMaxIter);
        cudaDeviceSynchronize();

        // File and console output
        writePGM(res.x, res.y, (char*)M, ".\\renders\\new_" + to_string(i));
        displayGreyscale(M, res.x, res.y, 4*40, 4 * 20);
    }




    // Free device memory
    cudaFree(M);
    cudaFree(blackDepth);
    cudaFree(nextMaxIter);


    return 0;
}