#ifndef SHADER_H
#define SHADER_H

// include glad to get all the required OpenGL headers
#include <glad/glad.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// useful to force stringify stuffs
// TODO: move me to a header or summink
// see: https://gcc.gnu.org/onlinedocs/gcc-13.2.0/cpp/Stringizing.html
// see: https://gcc.gnu.org/onlinedocs/gcc-13.2.0/cpp/Argument-Prescan.html
#define TO_STR(s) #s
#define X_TO_STR(X) TO_STR(X)

// TODO: rename me to shader program??
class Shader {
private:
    // utility function for checking shader compilation/linking errors.
    // ------------------------------------------------------------------------
    void checkCompileErrors(GLuint shader, std::string type);

public:
    // the shader program's handler ID
    GLuint shaderID;

    // constructor reads and builds the shader
    explicit Shader(const char* vertexPath, const char* fragmentPath);

    // use/activate the shader
    void glUseProgram();

    // glUniform setter,
    // but like... sould this really be const??
    template <typename T>
    void glUniform(const GLchar* name, T value) const;

    ~Shader();
};

// TODO: move move all of these to a cpp file and remove inline

template<>
inline void Shader::glUniform<GLint>(const GLchar* name, GLint value) const {
    glUniform1i(glGetUniformLocation(shaderID, name), value);
}

template<>
inline void Shader::glUniform<GLfloat>(const GLchar* name, GLfloat value) const {
    glUniform1f(glGetUniformLocation(shaderID, name), value);
}

inline Shader::Shader(const char* vertexShaderPath, const char* fragmentShaderPath) {
    //// 1. retrieve the vertex/fragment source code from filePath
    std::string vertexCode;
    std::string fragmentCode;
    std::ifstream vertexShaderFile;
    std::ifstream fragmentShaderFile;

    // ensure ifstream objects can throw exceptions:
    vertexShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fragmentShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    // TODO: nah this is kinda gross, should probably separate these,
    // since there's at least 2 completely separate causes for this to throw
    try {
        // open files
        vertexShaderFile.open(vertexShaderPath);
        fragmentShaderFile.open(fragmentShaderPath);
        std::stringstream vertexShaderStream, fragmentShaderStream;
        // read file's buffer contents into streams
        vertexShaderStream << vertexShaderFile.rdbuf();
        fragmentShaderStream << fragmentShaderFile.rdbuf();
        // close file handlers
        vertexShaderFile.close();
        fragmentShaderFile.close();
        // convert stream into string
        vertexCode = vertexShaderStream.str();
        fragmentCode = fragmentShaderStream.str();
    } catch (std::ifstream::failure& e) {
        std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << "\n";
    }

    const char* vertexShaderCode = vertexCode.c_str();
    const char* fragmentShaderCode = fragmentCode.c_str();

    //// 2. compile shaders
    unsigned int vertex, fragment;

    // vertex shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertexShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");

    // fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragmentShaderCode, nullptr);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");

    // shader Program
    shaderID = glCreateProgram();
    glAttachShader(shaderID, vertex);
    glAttachShader(shaderID, fragment);
    glLinkProgram(shaderID);
    checkCompileErrors(shaderID, "PROGRAM"); // is this redundant tho??

    // delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

// activate the shader
// ------------------------------------------------------------------------
inline void Shader::glUseProgram() {
    // global namespace operator lol, never knew about this
    ::glUseProgram(shaderID);
}

// TODO: noooooo, don't use strings
// enums plis
inline void Shader::checkCompileErrors(GLuint shader, std::string type) {
    GLint compileStatus;

    // TODO: store pointer during construction,
    // realloc only if we need more, using GL_INFO_LOG_LENGTH?
    GLchar infoLog[1024];

    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
    if (GL_FALSE == compileStatus) {
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);

        std::cerr << TO_STR(GL_COMPILE_STATUS)
            << ": " << compileStatus
            << " (==" TO_STR(GL_TRUE)
            << "==" X_TO_STR(GL_TRUE) " expected) "

            << "while compiling/linking: " << type << "\n"
            << infoLog
            << "===\n";
    }
}

inline Shader::~Shader() {
    glDeleteProgram(shaderID);
}

#endif
