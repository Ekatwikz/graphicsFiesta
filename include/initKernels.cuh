#ifndef INIT_KERNELS_CUH
#define INIT_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cameraStuff.cuh"

__global__ void initSpheres(Spheres* spheres, float3 centerMin, float3 centerMax, float radiusMin, float radiusMax, uint numSpheres, curandState_t* states) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < numSpheres) {
        curandState_t state = states[idx];
        spheres->centers[idx] = {
            (centerMin.x + curand_uniform(&state) * (centerMax.x - centerMin.x)) * (curand_uniform(&state) > 0.5F ? 1 : -1),
            (centerMin.y + curand_uniform(&state) * (centerMax.y - centerMin.y)) * (curand_uniform(&state) > 0.5F ? 1 : -1),
            (centerMin.z + curand_uniform(&state) * (centerMax.z - centerMin.z)) * (curand_uniform(&state) > 0.5F ? 1 : -1)
        };

        spheres->radii[idx] = radiusMin + curand_uniform(&state) * (radiusMax - radiusMin);

#ifdef PAUSE_FRAMES
        printf("[%d]: C_S:{%lf,%lf,%lf}|R:%lf\n", idx,
               spheres->centers[idx].x, spheres->centers[idx].y, spheres->centers[idx].z,
               spheres->radii[idx]);
#endif // PAUSE_FRAMES
    }
}

__global__ void initRand(unsigned int seed, curandState_t* states) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void initLights(Lights* lights, float3 centerMin, float3 centerMax,  uint numLights, curandState_t* states) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < numLights) {
        curandState_t state = states[idx];
        lights->centers[idx] = {
            (centerMin.x + curand_uniform(&state) * (centerMax.x - centerMin.x)) * (curand_uniform(&state) > 0.5F ? 1 : -1),
            (centerMin.y + curand_uniform(&state) * (centerMax.y - centerMin.y)) * (curand_uniform(&state) > 0.5F ? 1 : -1),
            (centerMin.z + curand_uniform(&state) * (centerMax.z - centerMin.z)) * (curand_uniform(&state) > 0.5F ? 1 : -1)
        };

        lights->colors[idx] = {
            curand_uniform(&state),
            curand_uniform(&state),
            curand_uniform(&state)
        };

#ifdef PAUSE_FRAMES
        printf("[%d]: C_L:{%lf,%lf,%lf}, C:{%f, %f, %f}\n", idx,
               lights->centers[idx].x, lights->centers[idx].y, lights->centers[idx].z,
               lights->colors[idx].x, lights->colors[idx].y, lights->colors[idx].z) ;
#endif // PAUSE_FRAMES
    }
}

#endif // INIT_KERNELS_CUH
