#version 330 core
out vec4 FragColor;

uniform float glfwTime;
in vec3 triangleColorOutput;

void main() {
	vec3 sinThingy = sin(glfwTime * vec3(4, 4.5, 5)) / 2 + 0.5;
	FragColor = vec4(triangleColorOutput * pow(sinThingy, vec3(0.7)), 1.0);
};
