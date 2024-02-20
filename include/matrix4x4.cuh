#ifndef MATRIX_4X4_CUH
#define MATRIX_4X4_CUH

#include <cuda_runtime.h>

struct Matrix4x4f {
    float4 rows[4];

    __device__ __host__ Matrix4x4f(const float4& row1, const float4& row2, const float4& row3, const float4& row4) {
        rows[0] = row1;
        rows[1] = row2;
        rows[2] = row3;
        rows[3] = row4;
    }

    __device__ __host__ Matrix4x4f() : Matrix4x4f{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}} {}

    __device__ __host__ Matrix4x4f(const float3& euler_angles, const float3& camera_position) : Matrix4x4f{} {
        // https://en.wikipedia.org/wiki/Rotation_matrix#General_3D_rotations
        float c_1 = cosf(euler_angles.x);
        float s_1 = sinf(euler_angles.x);

        float c_2 = cosf(euler_angles.y);
        float s_2 = sinf(euler_angles.y);

        float c_3 = cosf(euler_angles.z);
        float s_3 = sinf(euler_angles.z);

        rows[0] = make_float4(c_2 * c_3, s_1 * s_2 * s_3 - c_1 * s_3, c_1 * s_2 * c_3 + s_1 * s_3, camera_position.x);
        rows[1] = make_float4(c_2 * s_3, s_1 * s_2 * s_3 + c_1 * c_3, c_1 * s_2 * s_3 - s_1 * c_3, camera_position.y);
        rows[2] = make_float4(-s_2, s_1 * c_2, c_1 * c_2, camera_position.z);
    }

    __inline__ __device__ __host__ auto operator*(const float4& vec) const -> float4 {
        float4 result;
        result.x = rows[0].x * vec.x + rows[0].y * vec.y + rows[0].z * vec.z + rows[0].w * vec.w;
        result.y = rows[1].x * vec.x + rows[1].y * vec.y + rows[1].z * vec.z + rows[1].w * vec.w;
        result.z = rows[2].x * vec.x + rows[2].y * vec.y + rows[2].z * vec.z + rows[2].w * vec.w;
        result.w = rows[3].x * vec.x + rows[3].y * vec.y + rows[3].z * vec.z + rows[3].w * vec.w;
        return result;
    }

    __inline__  __device__ __host__ auto operator*(const float3& vec) const -> float3 {
        float4 result = *this * make_float4(vec.x, vec.y, vec.z, 1);
        return make_float3(result.x, result.y, result.z);
    }

    // b/c they're referring to different printfs?
#define PRINT_MATRIX() do { \
        printf("{"); \
        for (const auto& row : rows) { \
            printf("{%f %f %f %f},\n", row.x, row.y, row.z, row.w); \
        } \
        printf("}"); \
} while(0)

    __device__ void d_display() const { PRINT_MATRIX(); }

    __host__ void h_display() const { PRINT_MATRIX(); }
};

#endif //  MATRIX_4X4_CUH
