#ifndef CAMERASTUFF_HPP
#define CAMERASTUFF_HPP

#include <cuda_runtime.h>

#include "helper_cuda.h"

static const uint HALF_TURN_DEGREES = 90; // lol

class CameraInfo {
public:
    auto operator new(size_t amount) -> void* {
	CameraInfo* camInfo{};

	// I really need to check how the params of opnew are supposed to be used lol
	checkCudaErrors(cudaMallocManaged(&camInfo, sizeof(CameraInfo) * amount));

	memset(camInfo, 0, sizeof(CameraInfo));
	camInfo->fovDegrees = HALF_TURN_DEGREES;

	return camInfo;
    }

    void operator delete(void* cameraInfo) {
	checkCudaErrors(cudaFree(cameraInfo));
    }

    float3 center;
    uint2 imageResolution;
    float fovDegrees; // radians?
    float3 eulerAngles;
};

class Spheres {
public:
    auto operator new[](size_t amount) -> void* {
	Spheres* spheresInfo{};

	checkCudaErrors(cudaMallocManaged(&spheresInfo, sizeof(Spheres)));
	checkCudaErrors(cudaMalloc(&spheresInfo->centers, sizeof(float3) * amount));
	checkCudaErrors(cudaMalloc(&spheresInfo->radii, sizeof(float) * amount));

	spheresInfo->sz = amount;
	return spheresInfo;
    }

    void operator delete[](void* spheresV) {
	auto* spheres = static_cast<Spheres*>(spheresV);
	checkCudaErrors(cudaFree(spheres->centers));
	checkCudaErrors(cudaFree(spheres->radii));
	checkCudaErrors(cudaFree(spheres));
    }

    size_t sz;
    float3* centers;
    float* radii;
};

struct Lights {
public:
    auto operator new[](size_t amount) -> void* {
	Lights* lightsInfo{};

	checkCudaErrors(cudaMallocManaged(&lightsInfo, sizeof(Spheres)));
	checkCudaErrors(cudaMalloc(&lightsInfo->centers, sizeof(float3) * amount));
	checkCudaErrors(cudaMalloc(&lightsInfo->colors, sizeof(float) * amount));

	lightsInfo->sz = amount;
	return lightsInfo;
    }

    void operator delete[](void* lightsV) {
	auto* lights = static_cast<Lights*>(lightsV);
	checkCudaErrors(cudaFree(lights->centers));
	checkCudaErrors(cudaFree(lights->colors));
	checkCudaErrors(cudaFree(lights));
    }

    float3* centers;
    float3* colors;
    size_t sz;
    float3 attenuation;
    float ambientStrength;
};

#endif // CAMERASTUFF_HPP
