#include <cfloat>
#include "shaderProgram.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <cmath>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "helper_cuda.h"

#include "Keyboard.hpp"

constexpr uint CU_TEX_WIDTH = 1920;
constexpr uint CU_TEX_HEIGHT = 1080;

constexpr uint SPHERE_COUNT = 1000;
constexpr uint LIGHT_COUNT = 10;

constexpr float MOVE_SPEED = 0.3F;
constexpr float ROT_SPEED = 0.008F;

// callbacks
void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "Oops![0x%08X]: %s\n", error, description);
    glfwTerminate();
    exit(EXIT_FAILURE);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    (void)window;
    std::cerr << "Resized to: " << width << 'x' << height << '\n';
    glViewport(0, 0, width, height);
}

KeyboardState keyboardState;

void key_callback(GLFWwindow* window, int key, int scancode, int action,
                 int mods) {
    (void)scancode;
    (void)mods;

#ifdef DEBUG_KEYBOARD
    printf("Key:%d Action:%d\n", key, action);
#endif // DEBUG_KEYBOARD

    // close if Esc pressed
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else {
        keyboardState.handleKeyPress(key, action != GLFW_RELEASE);

#ifdef DEBUG_KEYBOARD
        float3 positionDelta = keyboardState.getPositionDelta(); 
        float3 eulerDelta = keyboardState.getEulerDelta(); 
        printf("Pos Delta: {%f, %f, %f}, Euler Delta: {%f, %f, %f}\n",
               positionDelta.x, positionDelta.y, positionDelta.z,
               eulerDelta.x, eulerDelta.y, eulerDelta.z
               );
#endif // DEBUG_KEYBOARD
    }
}

__device__ __host__ auto deg2rad(float degs) -> float {
    return degs * M_PI / 180;
}

using CameraInfo = struct CameraInfo_ {
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

__global__ void initRand(unsigned int seed, curandState_t* states) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

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
        //  https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
        float c_1 = cosf(euler_angles.x);
        float s_1 = sinf(euler_angles.x);
        float c_2 = cosf(euler_angles.y);
        float s_2 = sinf(euler_angles.y);
        float c_3 = cosf(euler_angles.z);
        float s_3 = sinf(euler_angles.z);

        rows[0] = make_float4(c_1 * c_2 * c_3 - s_1 * s_3, -c_3 * s_1 - c_1 * c_2 * s_3, c_1 * s_2, camera_position.x);
        rows[1] = make_float4(c_1 * s_3 + c_2 * c_3 * s_1, c_1 * c_3 - c_2 * s_1 * s_3, s_1 * s_2, camera_position.y);
        rows[2] = make_float4(-c_3 * s_2, s_2 * s_3, c_2, camera_position.z);
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
    return sqrt(dot(vec, vec));
}

auto __device__ __host__ normalize(const float3& vec) -> float3 {
    return vec / abs(vec);
}

// yoinked from here: https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml
// hopefully it just does what it says on the box (lhs reflected through rhs ig?) and I didn't mess up (💀💀)
auto __device__ __host__ reflect(const float3& lhs, const float3& rhs) -> float3 {
    return lhs - rhs * dot(lhs, rhs) * 2.0F;
}

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

#define GLCHECK() (__extension__({\
    GLenum glErrorVal; \
    const char* glErrorName; \
    while ((glErrorVal = glGetError()) != GL_NO_ERROR) { \
        switch (glErrorVal) { \
            case GL_INVALID_ENUM: \
                glErrorName = TO_STR(GL_INVALID_ENUM); \
                break; \
            case GL_INVALID_VALUE: \
                glErrorName = TO_STR(GL_INVALID_VALUE); \
                break; \
            case GL_INVALID_OPERATION: \
                glErrorName = TO_STR(GL_INVALID_OPERATION); \
                break; \
            /* case GL_STACK_OVERFLOW: \
                glErrorName = TO_STR(GL_STACK_OVERFLOW); \
                break; \
            case GL_STACK_UNDERFLOW: \
                glErrorName = TO_STR(GL_STACK_UNDERFLOW); \
                break; */ \
            case GL_OUT_OF_MEMORY: \
                glErrorName = TO_STR(GL_OUT_OF_MEMORY); \
                break; \
            case GL_INVALID_FRAMEBUFFER_OPERATION: \
                glErrorName = TO_STR(GL_INVALID_FRAMEBUFFER_OPERATION); \
                break; \
            /* case GL_CONTEXT_LOST: \
                 glErrorName = TO_STR(GL_CONTEXT_LOST); \
                 break; \
             case GL_TABLE_TOO_LARGE: \
                 break; \
                 glErrorName = TO_STR(GL_TABLE_TOO_LARGE); */ \
            default: \
                glErrorName = "???"; \
                break; \
        } \
        fprintf(stderr, __FILE__ ":%d in %s | glGetError()->0x%08X (%s)\n", __LINE__, static_cast<const char*>(__func__), glErrorVal, glErrorName); \
    } \
    glErrorVal; \
}))

auto main() -> int {
    // ===
    // === GLFW STUFFS
    // ===

    // set the error callback function for glfw stuff
    glfwSetErrorCallback(glfw_error_callback);

    // init glfw
    glfwInit();

    // hint that we'll use OpenGL 3.3 core? not sure exactly
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    // lole
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // create a window and context, check for errors,
    // make this window current
    GLFWwindow* window = glfwCreateWindow(
        CU_TEX_WIDTH, CU_TEX_HEIGHT, "IM GLing LESSGOOO", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    // init the glad loader or something, not sure
    if (gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)) ==
        0) {
        std::cerr << "Failed to initialize GLAD\n";
        return 1;
    }

    // set the viewport dimensions?
    glViewport(0, 0, CU_TEX_WIDTH, CU_TEX_HEIGHT);

    // set callbacks for window resizes and keystrokes
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);

    // ===
    // === SHADER STUFFS (Rect VAO, VBO, EBO)
    // ===
    float positions[] = {
        1.0F,  1.0F,  0.0F,  // top right
        1.0F,  -1.0F, 0.0F,  // bottom right
        -1.0F, -1.0F, 0.0F,  // bottom left
        -1.0F, 1.0F,  0.0F   // top left
    };

    float colors[] = {
        1.0F, 0.0F, 0.0F,  // top right
        0.0F, 1.0F, 0.0F,  // top right
        0.0F, 0.0F, 1.0F,  // top right
        1.0F, 1.0F, 1.0F   // top right
    };

    uint indices[] = {
        // note that we start from 0!
        0, 1, 3,  // first triangle
        1, 2, 3   // second triangle
    };

    // setup and bind vertex array object
    uint rectangle_VAO;
    glGenVertexArrays(1, &rectangle_VAO);
    glBindVertexArray(rectangle_VAO);

    // setup vertex buffer object
    // with our vertices
    uint rectangle_positions_VBO;
    glGenBuffers(1, &rectangle_positions_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, rectangle_positions_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions),
                 static_cast<GLvoid*>(positions), GL_STATIC_DRAW);

    // define the location and format of the vertex position attribute,
    // index=0, b/c we said location=0
    // 3 b/c 3 values,
    // GL_FALSE b/c we don't need normalization,
    // 3*floatsize is stride, (0 means packed, equivalent in this case)
    // first coord is at [0]
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // setup some colors too
    uint rectangle_colors_VBO;
    glGenBuffers(1, &rectangle_colors_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, rectangle_colors_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colors),
                 static_cast<GLvoid*>(colors), GL_STATIC_DRAW);

    // similar format for the colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // setup the element buffer object
    uint rectangle_points_EBO = 0;
    glGenBuffers(1, &rectangle_points_EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rectangle_points_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices),
                 static_cast<GLvoid*>(indices), GL_STATIC_DRAW);

    // enable the vertex attrib arrays?
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    ShaderProgram leShaderProgram{
        ShaderUnit<GL_FRAGMENT_SHADER>{File{"./shaders/fragment.glsl"}},
        ShaderUnit<GL_VERTEX_SHADER>{File{"./shaders/vertex.glsl"}}
    };

    // ===
    // === TEXTURE BINDING STUFFS (BOX)
    // ===
    unsigned int boxTexture;
    glGenTextures(1, &boxTexture);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, boxTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    {
        int width, height, nrChannels;
        stbi_set_flip_vertically_on_load(static_cast<int>(true));
        uint8_t* data = stbi_load("./textures/container.jpg", &width, &height, &nrChannels, 0);
        if (data != nullptr) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D); // probably not needed for later lol
        } else {
            std::cerr << "Failed to load texture\n";
        }

        stbi_image_free(data);
    }

    // ===
    // === TEXTURE BINDING STUFFS (SMILEY)
    // ===
    unsigned int smileyTexture;
    glGenTextures(1, &smileyTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, smileyTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    {
        int width, height, nrChannels;
        stbi_set_flip_vertically_on_load(static_cast<int>(true));
        uint8_t* data = stbi_load("./textures/awesomeface.png", &width, &height, &nrChannels, 0);
        if (data != nullptr) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D); // probably not needed for later lol
        } else {
            std::cerr << "Failed to load texture\n";
        }

        stbi_image_free(data);
    }

    // ===
    // === TEXTURE VBO STUFFS
    // ===
    float tex_coords[] = {
        1.0F,  1.0F, // top right
        1.0F,  0.0F, // bottom right
        0.0F, 0.0F, // bottom left
        0.0F, 1.0F, // top left
    };

    uint rectangle_tex_VBO;
    glGenBuffers(1, &rectangle_tex_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, rectangle_tex_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tex_coords),
                 static_cast<GLvoid*>(tex_coords), GL_STATIC_DRAW);

    // similar format for the colors
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(2);

    // ===
    // === CUDA TEXTURE STUFFS
    // ===

    // Create an OpenGL texture
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, CU_TEX_WIDTH, CU_TEX_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Register the texture with CUDA
    cudaGraphicsResource* cuda_texture_resource;
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_texture_resource, texture_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

    // install shader and set uniforms so we can tell samplers' offsets ig?
    leShaderProgram.glUseProgram();
    leShaderProgram.glUniform("box", 0);
    leShaderProgram.glUniform("smiley", 1);
    leShaderProgram.glUniform("cuda", 2);
    glUseProgram(0);

    // ===
    // === SPHERE / LIGHTS SETUP
    // ===
    Spheres* spheresInfo = nullptr;
    checkCudaErrors(cudaMallocManaged(&spheresInfo, sizeof(Spheres)));
    checkCudaErrors(cudaMalloc(&spheresInfo->centers, sizeof(float3) * SPHERE_COUNT));
    checkCudaErrors(cudaMalloc(&spheresInfo->radii, sizeof(float) * SPHERE_COUNT));
    Lights* lightsInfo = nullptr;
    checkCudaErrors(cudaMallocManaged(&lightsInfo, sizeof(Lights)));
    checkCudaErrors(cudaMalloc(&lightsInfo->centers, sizeof(float3) * LIGHT_COUNT));
    checkCudaErrors(cudaMalloc(&lightsInfo->colors, sizeof(float3) * LIGHT_COUNT));

    // Hand picked for decent (-ish) scaling up to 500 units, probably should tweak depending on how far out the lights might be
    lightsInfo->attenuation = {1, 0.001, 0.00002};
    lightsInfo->ambientStrength = 0.1 / LIGHT_COUNT;

    float3 centerMin = {5, 5, 5};
    float3 centerMax = {500, 500, 500};
    float radiusMin = 1;
    float radiusMax = 5;

    // Initialize cuRAND states
    curandState_t* sphereRandStates = nullptr;
    curandState_t* lightRandStates = nullptr;
    checkCudaErrors(cudaMalloc(&sphereRandStates, SPHERE_COUNT * sizeof(curandState_t)));
    checkCudaErrors(cudaMalloc(&lightRandStates, LIGHT_COUNT * sizeof(curandState_t)));

#ifdef RAND_SEED
    uint sphereSeed = RAND_SEED;
    uint lightSeed = RAND_SEED;
#else // !RAND_SEED
    uint sphereSeed = time(nullptr);
    uint lightSeed = time(nullptr) + 1;
#endif // RAND_SEED

    initRand<<<(SPHERE_COUNT + 255) / 256, 256>>>(sphereSeed, sphereRandStates); // preset seeds for easier debugging, for now
    initRand<<<(LIGHT_COUNT + 255) / 256, 256>>>(lightSeed, lightRandStates);
    checkCudaErrors(cudaDeviceSynchronize());

    // Initialize random spheres and lights
    initSpheres<<<(SPHERE_COUNT + 255) / 256, 256>>>(spheresInfo, centerMin, centerMax, radiusMin, radiusMax, SPHERE_COUNT, sphereRandStates);
    initLights<<<(LIGHT_COUNT + 255) / 256, 256>>>(lightsInfo, centerMin, centerMax, LIGHT_COUNT, lightRandStates);
    checkCudaErrors(cudaDeviceSynchronize());

    // ===
    // === Cam Setup
    // ===
    CameraInfo* camInfo = nullptr;

    checkCudaErrors(cudaMallocManaged(&camInfo, sizeof(CameraInfo)));
    memset(camInfo, 0, sizeof(CameraInfo));

    // TODO: get this from current reso or summink?
    camInfo->imageResolution = make_uint2(CU_TEX_WIDTH, CU_TEX_HEIGHT);
    camInfo->fovDegrees = 90;

    // ===
    // === RENDER LOOP
    // ===
    while (glfwWindowShouldClose(window) == 0) {
#ifdef DEBUG_TIMINGS
        cudaEvent_t stepStartEvent = nullptr;
        cudaEvent_t stepStopEvent = nullptr;
        float stepTime = 0;

        checkCudaErrors(cudaEventCreate(&stepStartEvent));
        checkCudaErrors(cudaEventCreate(&stepStopEvent));
        checkCudaErrors(cudaEventRecord(stepStartEvent));
#endif // DEBUG_TIMINGS

        auto glfwTime = static_cast<float>(glfwGetTime());

        glClearColor(0.2F, 0.3F, 0.3F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);

        // ===
        // ===
        // ===

        // Map the cuda texture to CUDA
        cudaArray* cuda_texture_array;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_texture_resource));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_texture_array, cuda_texture_resource, 0, 0));

         // Create a surface object
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_texture_array;
        cudaSurfaceObject_t output_surface;
        checkCudaErrors(cudaCreateSurfaceObject(&output_surface, &res_desc));

        // === Run the CUDA kernel
        dim3 block(16, 16);
        dim3 grid((CU_TEX_WIDTH + block.x - 1) / block.x, (CU_TEX_HEIGHT + block.y - 1) / block.y);

        write_texture_kernel<<<grid, block>>>(output_surface, camInfo, spheresInfo, SPHERE_COUNT, lightsInfo, LIGHT_COUNT);
        checkCudaErrors(cudaDeviceSynchronize());

        // Unmap the texture so that OpenGL can use it
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_texture_resource));

        // ===
        // ===
        // ===

        // install the shader program and draw stuffs
        leShaderProgram.glUseProgram();

        // pass time to the shaders so we can have a fiesta
        leShaderProgram.glUniform("glfwTime", glfwTime);

        glBindVertexArray(rectangle_VAO);
        // glDrawArrays(GL_TRIANGLES, 0, 3); // draw 3 verts
        glDrawElements(GL_TRIANGLES, sizeof(indices), GL_UNSIGNED_INT,
                       nullptr);  // draw using ebo
        glBindVertexArray(0);     // unbind, no need to unbind it every time tho

        glfwSwapBuffers(window);

#ifdef DEBUG_TIMINGS
        checkCudaErrors(cudaEventRecord(stepStopEvent));
        checkCudaErrors(cudaEventSynchronize(stepStopEvent));
        checkCudaErrors(cudaEventElapsedTime(&stepTime, stepStartEvent,
                                             stepStopEvent));
        std::fprintf(stderr, "%6.3fms\n", stepTime);
#endif // DEBUG_TIMINGS

        glfwPollEvents();
        camInfo->center += keyboardState.getPositionDelta() * MOVE_SPEED;
        camInfo->eulerAngles += keyboardState.getEulerDelta() * ROT_SPEED;

#ifdef PAUSE_FRAMES
        getchar(); // tmp boonk for going frame by frame
#endif // PAUSE_FRAMES
    }

    // cleanup a little and exit
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_texture_resource));
    glDeleteVertexArrays(1, &rectangle_VAO);
    glDeleteBuffers(1, &rectangle_positions_VBO);
    glDeleteBuffers(1, &rectangle_points_EBO);
    glfwDestroyWindow(window);
    glfwTerminate();
}
