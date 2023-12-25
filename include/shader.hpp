#ifndef SHADER_H
#define SHADER_H

// include glad to get all the required OpenGL headers
#include <glad/glad.h>

#include <iostream>

#include "tinyHelpers.h"

// TODO?: move this into the Shader class? idk
// PROGRAM isn't reeeeally a type, but whatev?
enum class ShaderType { PROGRAM, VERTEX, FRAGMENT };

class ShaderEntity {
private:
   GLuint shaderID;

public:
   // compiles/links the shader/program
   virtual auto setup() const -> void = 0;

   // helper function for checking shader compilation/linking errors.
   // TODO: change this to return a string
   auto checkBuildErrors() const -> void;

   [[nodiscard]] virtual auto glGetiv() const -> GLuint = 0;
   [[nodiscard]] virtual auto glGetInfoLog() const -> std::string = 0;

   [[nodiscard]] auto getShaderID() const -> GLuint { return shaderID; }

   ShaderEntity(const ShaderEntity&) = default;
   ShaderEntity(ShaderEntity&&) = delete;
   auto operator=(const ShaderEntity&) -> ShaderEntity& = default;
   auto operator=(ShaderEntity&&) -> ShaderEntity& = delete;

   // hmm
   virtual ~ShaderEntity() = 0;
};

auto ShaderEntity::checkBuildErrors() const -> void {
   ShaderType shaderType = ShaderType::PROGRAM; // TMP!!! just for compilation stuffs

   GLint compileStatus = 0;
   if (ShaderType::PROGRAM == shaderType) {
      glGetProgramiv(shaderID, GL_LINK_STATUS, &compileStatus);
   } else {
      glGetShaderiv(shaderID, GL_COMPILE_STATUS, &compileStatus);
   }

   if (GL_TRUE == compileStatus) {
      return;
   }

   // TODO: store pointer during construction,
   // realloc only if we need more, using GL_INFO_LOG_LENGTH?
   GLchar infoLog[1024] = {0};
   if (ShaderType::PROGRAM == shaderType) {
      glGetProgramInfoLog(shaderID, sizeof(infoLog), nullptr,
                          static_cast<GLchar*>(infoLog));
   } else {
      glGetShaderInfoLog(shaderID, sizeof(infoLog), nullptr,
                         static_cast<GLchar*>(infoLog));
   }

   std::cerr << (ShaderType::PROGRAM == shaderType ? TO_STR(GL_LINK_STATUS)
      : TO_STR(GL_COMPILE_STATUS))
      << ": " << compileStatus << " (==" TO_STR(GL_TRUE)
      << "==" X_TO_STR(GL_TRUE) " expected) "

      << "while compiling/linking: " << shaderType << "\n"
      << infoLog << "\n===\n";
}

// TODO: rename me to shader program??
class Shader {
   public:
    // use/activate the shader
    auto glUseProgram() const -> void;

    // glUniform setter,
    // but like... should this really be const??
    template <typename T>
    auto glUniform(const GLchar* name, T value) const -> void;

    // constructor reads and builds the shader
    explicit Shader(const char* vertexShaderPath,
                    const char* fragmentShaderPath);

    // no move, no copy, xdd
    Shader(const Shader&) = delete;
    Shader(Shader&&) = delete;
    auto operator=(const Shader&) -> Shader& = delete;
    auto operator=(Shader&&) -> Shader& = delete;
    ~Shader();

   private:
    // the shader program's handler ID
    GLuint shaderID;

    // utility function for checking shader compilation/linking errors.
    static auto checkCompileErrors(GLuint shader, ShaderType shaderType)
        -> void;
};

// maybe conversion op would be better? idk
static inline auto operator<<(std::ostream& outputStream,
                              ShaderType shaderTypeEnum) -> std::ostream& {
    // basic way to display enumz
    // a fancier way would be,
    // see: https://github.com/Neargye/magic_enum
    const char* ShaderTypeNames[]{TO_STR(ShaderType::PROGRAM),
                                  TO_STR(ShaderType::VERTEX),
                                  TO_STR(ShaderType::FRAGMENT)};

    return outputStream << ShaderTypeNames[static_cast<size_t>(shaderTypeEnum)];
}

#endif // SHADER_H
