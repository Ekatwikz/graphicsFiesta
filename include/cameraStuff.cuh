#ifndef CAMERASTUFF_HPP
#define CAMERASTUFF_HPP

#include <cuda_runtime.h>

#include "helper_cuda.h"

static const uint HALF_TURN_DEGREES = 90; // lol

class CameraInfo {
public:
    // idk this is pretty weird,
    // I'll probz think about a neat way of doin this at some point
    static auto alloc(uint width, uint height) -> CameraInfo* {
	CameraInfo* camInfo{};

	checkCudaErrors(cudaMallocManaged(&camInfo, sizeof(CameraInfo)));
	memset(camInfo, 0, sizeof(CameraInfo));

	// TODO: get this from current reso or summink?
	camInfo->imageResolution = make_uint2(width, height);
	camInfo->fovDegrees = HALF_TURN_DEGREES;

	return camInfo;
    }

    float3 center;
    uint2 imageResolution;
    float fovDegrees; // radians?
    float3 eulerAngles;
};

struct Spheres {
    float3* centers;
    float* radii;
};

struct Lights {
    float3* centers;
    float3* colors;
    float3 attenuation;
    float ambientStrength;
};

#endif // CAMERASTUFF_HPP
