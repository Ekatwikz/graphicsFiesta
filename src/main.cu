#include "shaderProgram.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "helper_cuda.h"

// mine stuffs:
#include "Keyboard.cuh"
#include "cameraStuff.cuh"
#include "initKernels.cuh"
#include "mathHelpers.cuh"

// my le raycasting
#include "raycastKernel.cuh"

constexpr uint CU_TEX_WIDTH = 1920;
constexpr uint CU_TEX_HEIGHT = 1080;

constexpr uint SPHERE_COUNT = 1000;
constexpr uint LIGHT_COUNT = 10;
 
constexpr float MOVE_SPEED = 0.3F;
constexpr float ROT_SPEED = 0.008F;

KeyboardState keyboardState;

void glfw_error_callback(int error, const char* description);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
auto glfwSetup() -> GLFWwindow*;

auto main() -> int {
    GLFWwindow* window{};
    if (nullptr == (window = glfwSetup())) {
        return 1;
    }

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
    uint rectangle_VAO{};
    glGenVertexArrays(1, &rectangle_VAO);
    glBindVertexArray(rectangle_VAO);

    // setup vertex buffer object
    // with our vertices
    uint rectangle_positions_VBO{};
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
    uint rectangle_colors_VBO{};
    glGenBuffers(1, &rectangle_colors_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, rectangle_colors_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colors),
                 static_cast<GLvoid*>(colors), GL_STATIC_DRAW);

    // similar format for the colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // setup the element buffer object
    uint rectangle_points_EBO{};
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
    uint boxTexture{};
    glGenTextures(1, &boxTexture);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, boxTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    {
        int width{};
        int height{};
        int nrChannels{};

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
    uint smileyTexture{};
    glGenTextures(1, &smileyTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, smileyTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    {
        int width{};
        int height{};
        int nrChannels{};

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

    uint rectangle_tex_VBO{};
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
    GLuint texture_id{};
    glGenTextures(1, &texture_id);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, CU_TEX_WIDTH, CU_TEX_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Register the texture with CUDA
    cudaGraphicsResource* cuda_texture_resource{};
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
    CameraInfo* camInfo{};
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
        cudaArray* cuda_texture_array{};
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_texture_resource));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_texture_array, cuda_texture_resource, 0, 0));

         // Create a surface object
        cudaResourceDesc res_desc{};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_texture_array;
        cudaSurfaceObject_t output_surface{};
        checkCudaErrors(cudaCreateSurfaceObject(&output_surface, &res_desc));

        // === Run the CUDA kernel
        dim3 block(16, 16);
        dim3 grid((CU_TEX_WIDTH + block.x - 1) / block.x, (CU_TEX_HEIGHT + block.y - 1) / block.y);

        raycast_kernel<<<grid, block>>>(output_surface, camInfo, spheresInfo, SPHERE_COUNT, lightsInfo, LIGHT_COUNT);
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

auto glfwSetup() -> GLFWwindow* {
    // set the error callback function for glfw stuff
    glfwSetErrorCallback(glfw_error_callback);

    // init glfw
    glfwInit();

    // hint that we'll use OpenGL 3.3 core? not sure exactly
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    // not sure if this is even enough
    // with all the weird stuff I've done anyway lole
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // create a window and context, check for errors,
    // make this window current
    GLFWwindow* window = glfwCreateWindow(
        CU_TEX_WIDTH, CU_TEX_HEIGHT, "IM GLing LESSGOOO", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);

    // init the glad loader or something, not sure
    if (gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)) == 0) {
        std::cerr << "Failed to initialize GLAD\n";
        return nullptr;
    }

    // set the viewport dimensions?
    glViewport(0, 0, CU_TEX_WIDTH, CU_TEX_HEIGHT);

    // set callbacks for window resizes and keystrokes
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);

    return window;
}

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
