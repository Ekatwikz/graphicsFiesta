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

// PROGRAM isn't reeeeally a type, but whatev?
enum class ShaderType{ PROGRAM, VERTEX, FRAGMENT };

// maybe conversion op would be better? idk
static inline auto operator<<(std::ostream& outputStream, ShaderType shaderTypeEnum)
-> std::ostream& {
    // basic way to display enumz
    // a fancier way would be,
    // see: https://github.com/Neargye/magic_enum
    const char* ShaderTypeNames[] {
        TO_STR(ShaderType::PROGRAM),
        TO_STR(ShaderType::VERTEX),
        TO_STR(ShaderType::FRAGMENT)
    };

    return outputStream << ShaderTypeNames[static_cast<size_t>(shaderTypeEnum)];
}

// TODO: rename me to shader program??
class Shader {
private:
    // utility function for checking shader compilation/linking errors.
    // ------------------------------------------------------------------------
    static auto checkCompileErrors(GLuint shader, ShaderType shaderType);

public:
    // the shader program's handler ID
    GLuint shaderID;

    // constructor reads and builds the shader
    explicit Shader(const char* vertexShaderPath, const char* fragmentShaderPath);

    // use/activate the shader
    auto glUseProgram() const;

    // glUniform setter,
    // but like... sould this really be const??
    template <typename T>
    auto glUniform(const GLchar* name, T value) const;

    // no move, no copy, xdd
    Shader(const Shader&)=delete;
    Shader(Shader&&) = delete;
    auto operator=(const Shader&) -> Shader& = delete;
    auto operator=(Shader&&) -> Shader& = delete;
    ~Shader();
};

// TODO: move move all of these to a cpp file and remove inline

template<>
inline auto Shader::glUniform<GLint>(const GLchar* name, GLint value) const {
    glUniform1i(glGetUniformLocation(shaderID, name), value);
}

template<>
inline auto Shader::glUniform<GLfloat>(const GLchar* name, GLfloat value) const {
    glUniform1f(glGetUniformLocation(shaderID, name), value);
}

// activate the shader
// ------------------------------------------------------------------------
inline auto Shader::glUseProgram() const {
    // global namespace operator lol, never knew about this
    ::glUseProgram(shaderID);
}

inline auto Shader::checkCompileErrors(GLuint shader, ShaderType shaderType) {
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
        glGetProgramInfoLog(shader, sizeof(infoLog), nullptr, static_cast<GLchar*>(infoLog));
    } else {
        glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, static_cast<GLchar*>(infoLog));
    }

    std::cerr
        << (ShaderType::PROGRAM == shaderType
        ? TO_STR(GL_LINK_STATUS)
        : TO_STR(GL_COMPILE_STATUS))
        << ": " << compileStatus
        << " (==" TO_STR(GL_TRUE)
        << "==" X_TO_STR(GL_TRUE) " expected) "

        << "while compiling/linking: " << shaderType << "\n"
        << infoLog
        << "===\n";
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
        std::stringstream vertexShaderStream;
        std::stringstream fragmentShaderStream;

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
    unsigned int vertex = 0;
    unsigned int fragment = 0;

    // vertex shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertexShaderCode, nullptr);
    glCompileShader(vertex);
    checkCompileErrors(vertex, ShaderType::VERTEX);

    // fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragmentShaderCode, nullptr);
    glCompileShader(fragment);
    checkCompileErrors(fragment, ShaderType::FRAGMENT);

    // shader Program
    shaderID = glCreateProgram();
    glAttachShader(shaderID, vertex);
    glAttachShader(shaderID, fragment);
    glLinkProgram(shaderID);
    checkCompileErrors(shaderID, ShaderType::PROGRAM); // is this redundant tho??

    // delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

inline Shader::~Shader() {
    glDeleteProgram(shaderID);
}

#endif
