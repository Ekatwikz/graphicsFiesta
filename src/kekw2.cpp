#define GLFW_INCLUDE_NONE

#include <GLFW/glfw3.h>

#include <cmath>

#include "shaderProgram.hpp"

// callbacks
void error_callback(int error, const char* description) {
    fprintf(stderr, "Oops![0x%08X]: %s\n", error, description);
    glfwTerminate();
    exit(EXIT_FAILURE);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    (void)window;
    glViewport(0, 0, width, height);
}

void key_handler(GLFWwindow* window, int key, int scancode, int action,
                 int mods) {
    (void)scancode;
    (void)mods;

    printf("Key:%d Action:%d\n", key, action);

    // close if Esc pressed
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

constexpr uint SCR_WIDTH = 800;
constexpr uint SCR_HEIGHT = 600;
auto main() -> int {
    // set the error callback function for glfw stuff
    glfwSetErrorCallback(error_callback);

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
        SCR_WIDTH, SCR_HEIGHT, "IM GLing LESSGOOO", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    // init the glad loader or something, not sure
    if (gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)) ==
        0) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return 1;
    }

    // set the viewport dimensions?
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

    // set callbacks for window resizes and keystrokes
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_handler);

    float positions[] = {
        0.5F,  0.5F,  0.0F,  // top right
        0.5F,  -0.5F, 0.0F,  // bottom right
        -0.5F, -0.5F, 0.0F,  // bottom left
        -0.5F, 0.5F,  0.0F   // top left
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
    uint rectangle_VAO = 0;
    glGenVertexArrays(1, &rectangle_VAO);
    glBindVertexArray(rectangle_VAO);

    // setup vertex buffer object
    // with our vertices
    uint rectangle_positions_VBO = 0;
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
    uint rectangle_colors_VBO = 0;
    glGenBuffers(1, &rectangle_colors_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, rectangle_colors_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions),
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

    // construct shader program from vertex and fragment
    ShaderProgram leShaderProgram{"./shaders/vertex.glsl", "./shaders/fragment.glsl"};

    // render loop
    while (glfwWindowShouldClose(window) == 0) {
        glClearColor(0.2F, 0.3F, 0.3F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);

        // install the shader program and draw stuffs
        leShaderProgram.glUseProgram();

        // vary the shape's color using the uniform in the fragshader
        leShaderProgram.glUniform("glfwTime",
                                  static_cast<float>(glfwGetTime()));

        glBindVertexArray(rectangle_VAO);
        // glDrawArrays(GL_TRIANGLES, 0, 3); // draw 3 verts
        glDrawElements(GL_TRIANGLES, sizeof(indices), GL_UNSIGNED_INT,
                       nullptr);  // draw using ebo
        glBindVertexArray(0);     // unbind, no need to unbind it every time tho

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // cleanup a little and exit
    glDeleteVertexArrays(1, &rectangle_VAO);
    glDeleteBuffers(1, &rectangle_positions_VBO);
    glDeleteBuffers(1, &rectangle_points_EBO);
    glfwTerminate();
}
