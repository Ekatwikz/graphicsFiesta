#version 330 core
out vec4 FragColor;

uniform float glfwTime;
in vec3 triangleColorOutput;
in vec2 texCoord;

uniform sampler2D box;
uniform sampler2D smiley;

void main() {
	vec3 sinThingy = sin(glfwTime * vec3(4, 4.5, 5)) / 2 + 0.5;
	vec4 boonkColorDance = vec4(triangleColorOutput * pow(sinThingy, vec3(0.7)), 1.0);

	FragColor = mix(texture(box, texCoord), texture(smiley, vec2(1 - texCoord.x, texCoord.y)), sinThingy.x) * boonkColorDance;
	//FragColor = texture(box, texCoord);
}
