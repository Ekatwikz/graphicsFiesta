#ifndef CAMERASTUFF_HPP
#define CAMERASTUFF_HPP

#include <cuda_runtime.h>

struct CameraInfo {
    float3 center;
    uint2 imageResolution;
    float fovDegrees; // radians?
    float3 eulerAngles;
};

// SOA type stuff ig?
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
