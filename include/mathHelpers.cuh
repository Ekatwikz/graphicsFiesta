#ifndef MATH_HELPERS_H
#define MATH_HELPERS_H

#include <cuda_runtime.h>

// little stuffs to make fiddling with meths (mostly float3) easier

auto __device__ __host__ deg2rad(float degs) -> float {
    constexpr uint8_t HALF_TURN_DEGS = 180;
    return static_cast<float>(degs * M_PI / HALF_TURN_DEGS);
}

auto __device__ __host__ operator-(const float3& lhs, const float3& rhs) -> float3 {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

auto __device__ __host__ operator+(const float3& lhs, const float3& rhs) -> float3 {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

auto __device__ __host__ operator+=(float3& lhs, const float3& rhs) -> float3 {
    return lhs = lhs + rhs;
}

auto __device__ __host__ dot(const float3& lhs, const float3& rhs) -> float {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

// pure spaghetti code, yikes
// TODO: template-ize this crap
auto __device__ __host__ clamp(float3& vec, float min, float max) -> float3 {
    if (vec.x < min) {
        vec.x = min;
    } else if (vec.x > max) {
        vec.x = max;
    }

    if (vec.y < min) {
        vec.y = min;
    } else if (vec.y > max) {
        vec.y = max;
    }

    if (vec.z < min) {
        vec.z = min;
    } else if (vec.z > max) {
        vec.z = max;
    }

    return vec;
}

auto __device__ __host__ operator/(const float3& lhs, float rhs) -> float3 {
    return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
}

auto __device__ __host__ operator*(const float3& lhs, float rhs) -> float3 {
    return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}

auto __device__ __host__ operator-(const float3& vec) -> float3 {
    return vec * -1.0F;
}

auto __device__ __host__ abs(const float3& vec) -> float {
    return sqrtf(dot(vec, vec));
}

auto __device__ __host__ normalize(const float3& vec) -> float3 {
    return vec / abs(vec);
}

// yoinked from here: https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml
// hopefully it just does what it says on the box (lhs reflected through rhs ig?) and I didn't mess up (💀💀)
auto __device__ __host__ reflect(const float3& lhs, const float3& rhs) -> float3 {
    return lhs - rhs * dot(lhs, rhs) * 2.0F;
}

#endif // MATH_HELPERS_H
