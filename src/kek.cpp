#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>

void error_callback(int error, const char* description) {
    (void)error;

    fprintf(stderr, "Oops!: %s\n", description);
    glfwTerminate();
    exit(EXIT_FAILURE);
}

auto main() -> int {
    glfwSetErrorCallback(error_callback);
    if (glfwInit() == 0) {
        // GLFW init failed
        std::cerr << "bruh\n";
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window =
        glfwCreateWindow(640, 480, "pep egah", nullptr, nullptr);
    if (window == nullptr) {
        // Window or OpenGL context creation failed
        std::cerr << "bruh\n";
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}
