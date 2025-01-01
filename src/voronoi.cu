#include <stdio.h>

#include "shared_with_cuda.h"

__global__ void cuda_voronoi_kernel(Point *points, int pointc, int *indices, int w, int h)
{
    // threadIdx.x is thread id 
    //  blockDim.x is number of threads

    int start_column = (w/blockDim.x) * threadIdx.x;
    int   end_column = (w/blockDim.x) * (threadIdx.x+1);
    printf("(%d/%d) running columns %d-%d\n", threadIdx.x, blockDim.x, start_column, end_column);

    for (int i = start_column; i < end_column; i += 1)
    {
        for (int j = 0; j < h; j += 1)
        {
            float minimum_distance = 100000.0f;
            int minimum_index = 0;

            for (int k = 0; k < pointc; ++k)
            {
                // Get distance from each point
                float distance = 0.0f;
                float x_distance = points[k].x - i;
                float y_distance = points[k].y - j;
                distance = (x_distance * x_distance) + (y_distance * y_distance); // Skipping the sqrt here.

                if (distance < minimum_distance)
                {
                    minimum_distance = distance;
                    minimum_index = k;
                }
            }

            int index = j + i*h;
            indices[index] = minimum_index;
        }
    }
}

extern "C" void cuda_voronoi(Point *points, int pointc, int *indices, int w, int h)
{
    Point *device_points;
    int *device_indices;

    cudaMalloc((void**)(&device_indices), sizeof(int)*w*h);

    cudaMalloc((void**)(&device_points), sizeof(Point)*pointc);
    cudaMemcpy(device_points, points, sizeof(Point)*pointc, cudaMemcpyHostToDevice);

    cuda_voronoi_kernel<<<1,128>>>(device_points, pointc, device_indices, w, h);

    cudaMemcpy(indices, device_indices, sizeof(int)*w*h, cudaMemcpyDeviceToHost);

    cudaFree(device_points);
    cudaFree(device_indices);
}
