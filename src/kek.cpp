#include <cstdlib>
#include <iostream>
#include <GLFW/glfw3.h>

void error_callback(int error, const char* description) {
	(void) error;

	fprintf(stderr, "Oops!: %s\n", description);
	glfwTerminate();
	exit(EXIT_FAILURE);
}

int main () {
	glfwSetErrorCallback(error_callback);
	if (!glfwInit()) {
		// GLFW init failed
		std::cerr << "bruh\n";
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	GLFWwindow* window = glfwCreateWindow(640, 480, "pep egah", NULL, NULL);
	if (!window) {
		// Window or OpenGL context creation failed
		std::cerr << "bruh\n";
	}

	glfwDestroyWindow(window);
	glfwTerminate();
}
