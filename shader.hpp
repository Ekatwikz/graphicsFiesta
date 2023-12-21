#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h> // include glad to get all the required OpenGL headers

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// TODO: rename me to shader program??
class Shader {
private:
    // utility function for checking shader compilation/linking errors.
    // ------------------------------------------------------------------------
    void checkCompileErrors(unsigned int shader, std::string type);

public:
    // the program ID
    uint shaderID;

    // constructor reads and builds the shader
    explicit Shader(const char* vertexPath, const char* fragmentPath);

    // use/activate the shader
    void use(); // operator() ?

    // glUniform setter,
    // but like... is this really const??
    template <typename T>
    void glUniform(const GLchar* name, T value) const;

    ~Shader();
};

// TODO: move move all of these to a cpp file and remove inline

template<>
inline void Shader::glUniform<int>(const GLchar* name, int value) const {
    glUniform1i(glGetUniformLocation(shaderID, name), value);
}

template<>
inline void Shader::glUniform<float>(const GLchar* name, float value) const {
    glUniform1f(glGetUniformLocation(shaderID, name), value);
}

inline Shader::Shader(const char* vertexPath, const char* fragmentPath) {
    //// 1. retrieve the vertex/fragment source code from filePath
    std::string vertexCode;
    std::string fragmentCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;

    // ensure ifstream objects can throw exceptions:
    vShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
    try {
        // open files
        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);
        std::stringstream vShaderStream, fShaderStream;
        // read file's buffer contents into streams
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();
        // close file handlers
        vShaderFile.close();
        fShaderFile.close();
        // convert stream into string
        vertexCode = vShaderStream.str();
        fragmentCode = fShaderStream.str();
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
    checkCompileErrors(shaderID, "PROGRAM");

    // delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

// activate the shader
// ------------------------------------------------------------------------
inline void Shader::use() {
    glUseProgram(shaderID);
}

inline void Shader::checkCompileErrors(uint shader, std::string type) {
    int success;
    char infoLog[1024];

    // TODO: rearrange this cringe shit
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog <<
                "\n -- --------------------------------------------------- -- " << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog <<
                "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}

inline Shader::~Shader() {
    glDeleteProgram(shaderID);
}

#endif
