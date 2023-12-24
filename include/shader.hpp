#ifndef SHADER_H
#define SHADER_H

// include glad to get all the required OpenGL headers
#include <glad/glad.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "tinyHelpers.h"

// TODO?: move this into the Shader class? idk
// PROGRAM isn't reeeeally a type, but whatev?
enum class ShaderType{ PROGRAM, VERTEX, FRAGMENT };

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
    auto glUseProgram() const -> void;

    // glUniform setter,
    // but like... sould this really be const??
    template <typename T>
    auto glUniform(const GLchar* name, T value) const -> void;

    // no move, no copy, xdd
    Shader(const Shader&) = delete;
    Shader(Shader&&) = delete;
    auto operator=(const Shader&) -> Shader& = delete;
    auto operator=(Shader&&) -> Shader& = delete;
    ~Shader();
};

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

#endif
