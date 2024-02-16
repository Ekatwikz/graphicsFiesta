#ifndef RAYCAST_KERNEL_CUH
#define RAYCAST_KERNEL_CUH

#include <cfloat>
#include <cuda_runtime.h>

#include "cameraStuff.cuh"
#include "matrix4x4.cuh"
#include "mathHelpers.cuh"

__global__ void write_texture_kernel(cudaSurfaceObject_t output_surface, CameraInfo* camInfo,
                                     Spheres* spheresInfo, uint sphereCount,
                                     Lights* lightsInfo, uint lightsCount) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    uint width = camInfo->imageResolution.x;
    uint height = camInfo->imageResolution.y;

    float fovScale = tan(deg2rad(camInfo->fovDegrees / 2));

    if (x < width && y < height) {
        uint2 pixel = make_uint2(x, y);
        double2 pixelNDC = make_double2((pixel.x + 0.5) / width, (pixel.y + 0.5) / height);

        float aspectRatio = 1.0F * width / height;
        float2 pixCam = make_float2((2 * pixelNDC.x - 1) * aspectRatio * fovScale,
                                      (1 - 2 * pixelNDC.y) * fovScale);
        float3 pixCamCoord = make_float3(pixCam.x, pixCam.y, -1);

        Matrix4x4f camToWorld{camInfo->eulerAngles, camInfo->center};
        float3 pixWorldCoord = camToWorld * pixCamCoord;
        float3 rayDirection = pixWorldCoord - camInfo->center;
        float3 D = normalize(rayDirection);

#ifdef PAUSE_FRAMES
        //camToWorld.d_display();
        printf("Pix:{%d,%d}|NDC:{%lf,%lf}|Cam:{%lf,%lf}|World:{%lf,%lf,%lf}|Dir:{%lf,%lf,%lf}\n",
               pixel.x, pixel.y,
               pixelNDC.x, pixelNDC.y,
               pixCam.x, pixCam.y,
               pixWorldCoord.x, pixWorldCoord.y, pixWorldCoord.z,
               rayDirection.x, rayDirection.y, rayDirection.z);
#endif // PAUSE_FRAMES

        float t_hc = NAN;
        float t_0 = FLT_MAX;
        long intersectedIndex = -1; // will just use -1 for "no intersection"

        for (uint i = 0; i < sphereCount; ++i) {
            // Geometric solution from:
            // https://github.com/scratchapixel/website/blob/main/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.md?plain=1
            float currentRadius = spheresInfo->radii[i];

            float3 L = spheresInfo->centers[i] - camInfo->center;
            float t_ca = dot(L, D);
            if (t_ca < 0) { // sphere is behind
                continue;
            }

            float d = sqrt(dot(L, L) - t_ca * t_ca);
            if (d > spheresInfo->radii[i]) { // ray "missed"
                continue;
            }

            float curr_t_hc = sqrt(currentRadius * currentRadius - d * d);
            float curr_t_0 = t_ca - curr_t_hc;
            if (curr_t_0 > t_0) { // blocked by some closer sphere
                continue;
            }

            t_hc = curr_t_hc;
            intersectedIndex = i;
            t_0 = curr_t_0;
        }

        // === DRAW STUFFS ===
        uchar4 tmpPixData = make_uchar4(0, 0, 0, 255);

        if (-1 != intersectedIndex) { // if we're lookin at somethin
            // Lighting Setup from here:
            // https://learnopengl.com/Lighting/Basic-Lighting
            float3 intersectionPoint = D * t_0 + camInfo->center;
            float3 intersectionNormal = normalize(intersectionPoint - spheresInfo->centers[intersectedIndex]);

#ifdef PAUSE_FRAMES
            printf("%d,%d: t_hc:%lf t_0:%f r:%f [%ld] {%f, %f, %f}->{%f, %f, %f} (C_S:{%f, %f, %f})\n", x, y, t_hc, t_0,
                   spheresInfo->radii[intersectedIndex], intersectedIndex,
                   pixWorldCoord.x, pixWorldCoord.y, pixWorldCoord.z,
                   intersectionPoint.x, intersectionPoint.y, intersectionPoint.z,
                   spheresInfo->centers[intersectedIndex].x, spheresInfo->centers[intersectedIndex].y, spheresInfo->centers[intersectedIndex].z);
#endif // PAUSE_FRAMES

            float3 ambient = {0, 0, 0};
            float3 diffuse = {0, 0, 0};
            float3 specular = {0, 0, 0};
            for(uint i = 0; i < lightsCount; ++i) {
                float3 lightPos = lightsInfo->centers[i];
                float3 lightColor = lightsInfo->colors[i];
                float3 lightVec = lightPos - intersectionPoint;
                float3 lightDir = normalize(lightVec);
                float lightDistance = abs(lightVec);

                ambient += lightColor * lightsInfo->ambientStrength;

                float attenuation = 1 / (lightsInfo->attenuation.x
                    + lightsInfo->attenuation.y * lightDistance
                    + lightsInfo->attenuation.z * lightDistance * lightDistance);

                float diffuseStrength = max(dot(intersectionNormal, lightDir), 0.0F);
                diffuse += lightColor * diffuseStrength * attenuation;

                float specularIntensity = 0.5;
                float3 viewDir = normalize(camInfo->center - intersectionPoint);
                float3 reflectDir = reflect(-lightDir, intersectionNormal);
                float shininess = 32; // TODO: move this somewhere else?
                auto specularStrength = static_cast<float>(pow( max(dot(viewDir, reflectDir), 0.0F), shininess));
                specular += lightColor * specularIntensity * specularStrength * attenuation;
            }

            float3 color = (ambient + diffuse + specular) * 255;
            color = clamp(color, 0, 255); // just in case lol, idk

#ifdef PAUSE_FRAMES
            printf("A:{%f, %f, %f}|D:{%f, %f, %f}|S:{%f, %f, %f}|C:{%f, %f, %f}\n",
                   ambient.x, ambient.y, ambient.z,
                   diffuse.x, diffuse.y, diffuse.z,
                   specular.x, specular.y, specular.z,
                   color.x, color.y, color.z);
#endif // PAUSE_FRAMES

            tmpPixData = make_uchar4(color.x, color.y, color.z, 255);
        }

        // manually flip texture at the last moment, I dont remember why but opengl goofs up the texture otherwise
        surf2Dwrite(tmpPixData, output_surface, x * sizeof(uchar4), height - 1 - y);
    }
}

#endif // RAYCAST_KERNEL_CUH
