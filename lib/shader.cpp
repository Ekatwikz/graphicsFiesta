#include "shader.hpp"

#include "file.hpp"

Shader::Shader(const char* vertexShaderPath, const char* fragmentShaderPath) {
    File vertexShader;
    try {
        vertexShader = File{vertexShaderPath};
    } catch (std::ifstream::failure& ex) {
        std::cerr << "Oops! IO exception when constructing vertex shader\n"
                  << ex.what() << "\n===\n";
    }

    File fragmentShader;
    try {
        fragmentShader = File{fragmentShaderPath};
    } catch (std::ifstream::failure& ex) {
        std::cerr << "Oops! IO exception when constructing fragment shader\n"
                  << ex.what() << "\n===\n";
    }

    // vertex shader
    uint vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, vertexShader.getRawContentP(), nullptr);
    glCompileShader(vertex);
    checkCompileErrors(vertex, ShaderType::VERTEX);

    // fragment Shader
    uint fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, fragmentShader.getRawContentP(), nullptr);
    glCompileShader(fragment);
    checkCompileErrors(fragment, ShaderType::FRAGMENT);

    // shader Program
    shaderID = glCreateProgram();
    glAttachShader(shaderID, vertex);
    glAttachShader(shaderID, fragment);

    glLinkProgram(shaderID);
    checkCompileErrors(shaderID, ShaderType::PROGRAM);

    // delete the shaders as they're linked into our program now and no longer
    // necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

Shader::~Shader() { glDeleteProgram(shaderID); }

auto Shader::glUseProgram() const -> void {
    // global namespace operator lul
    ::glUseProgram(shaderID);
}

template <>
auto Shader::glUniform<GLint>(const GLchar* name, GLint value) const -> void {
    glUniform1i(glGetUniformLocation(shaderID, name), value);
}

template <>
auto Shader::glUniform<GLfloat>(const GLchar* name, GLfloat value) const
    -> void {
    glUniform1f(glGetUniformLocation(shaderID, name), value);
}

auto Shader::checkCompileErrors(GLuint shader, ShaderType shaderType) -> void {
    GLint compileStatus = 0;
    if (ShaderType::PROGRAM == shaderType) {
        glGetProgramiv(shader, GL_LINK_STATUS, &compileStatus);
    } else {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
    }

    if (GL_TRUE == compileStatus) {
        return;
    }

    // TODO: store pointer during construction,
    // realloc only if we need more, using GL_INFO_LOG_LENGTH?
    GLchar infoLog[1024] = {0};
    if (ShaderType::PROGRAM == shaderType) {
        glGetProgramInfoLog(shader, sizeof(infoLog), nullptr,
                            static_cast<GLchar*>(infoLog));
    } else {
        glGetShaderInfoLog(shader, sizeof(infoLog), nullptr,
                           static_cast<GLchar*>(infoLog));
    }

    std::cerr << (ShaderType::PROGRAM == shaderType ? TO_STR(GL_LINK_STATUS)
                                                    : TO_STR(GL_COMPILE_STATUS))
              << ": " << compileStatus << " (==" TO_STR(GL_TRUE)
              << "==" X_TO_STR(GL_TRUE) " expected) "

              << "while compiling/linking: " << shaderType << "\n"
              << infoLog << "\n===\n";
}
