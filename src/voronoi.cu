
#include <stdio.h>
#include <float.h>

#include "shared_with_cuda.h"

__global__ void voronoi_cuda_kernel(Point *points, char *pixels)
{
    // threadIdx.x is thread id 
    //  blockDim.x is number of threads

    int pixels_per_thread = SCREEN_W / (gridDim.x * blockDim.x);
    int start_column = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int   end_column = start_column + pixels_per_thread;

    for (int i = start_column; i < end_column; i += 1)
    {
        for (int j = 0; j < SCREEN_H; j += 1)
        {
            float minimum_distance = FLT_MAX;
            int minimum_index = 0;

            for (int k = 0; k < POINT_COUNT; k += 1)
            {
                // Get distance from each point
                float distance = 0.0f;
                float x_distance = points[k].x - i;
                float y_distance = points[k].y - j;
                distance = (x_distance * x_distance) + (y_distance * y_distance);

                if (distance < minimum_distance)
                {
                    minimum_distance = distance;
                    minimum_index = k;
                }
            }

            int offset = (j*4) * SCREEN_W + (i*4);
            pixels[offset + 0] = (char)points[minimum_index].b;
            pixels[offset + 1] = (char)points[minimum_index].g;
            pixels[offset + 2] = (char)points[minimum_index].r;
            pixels[offset + 3] = (char)255;
        }
    }
}

extern "C" void voronoi_cuda(Point *points, char *pixels)
{
    Point *device_points;
    char *device_pixels;

    cudaMalloc((void**)(&device_pixels), sizeof(char)*4*SCREEN_W*SCREEN_H);

    cudaMalloc((void**)(&device_points), sizeof(Point)*POINT_COUNT);
    cudaMemcpy(device_points, points, sizeof(Point)*POINT_COUNT, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int block_count = (SCREEN_W + threads_per_block - 1) / threads_per_block;

    voronoi_cuda_kernel<<<block_count, threads_per_block>>>(device_points, device_pixels);

    cudaMemcpy(pixels, device_pixels, sizeof(char)*4*SCREEN_W*SCREEN_H, cudaMemcpyDeviceToHost);

    cudaFree(device_points);
    cudaFree(device_pixels);
}

