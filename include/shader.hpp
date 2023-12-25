#ifndef SHADER_H
#define SHADER_H

// include glad to get all the required OpenGL headers
#include <glad/glad.h>

#include <iostream>

#include "tinyHelpers.h"

// TODO?: move this into the Shader class? idk
// PROGRAM isn't reeeeally a type, but whatev?
enum class ShaderType { PROGRAM, VERTEX, FRAGMENT };
static inline auto operator<<(std::ostream& outputStream,
                              ShaderType shaderTypeEnum) -> std::ostream&;

class ShaderEntity {
public:
   // compiles/links the shader/program
   virtual auto setup() const -> void = 0;

   // helper function for checking shader compilation/linking errors.
   // TODO: change this to return a string
   auto displaySetupErrors() const -> void;

   // checks if last compile/link was successful
   [[nodiscard]] virtual auto glGetSetupiv() const -> GLint = 0;

   [[nodiscard]] virtual auto glGetInfoLog() const -> std::string = 0;

   [[nodiscard]] auto getShaderID() const -> GLuint { return shaderID; }

   ShaderEntity(const ShaderEntity&) = default;
   ShaderEntity(ShaderEntity&&) = delete;
   auto operator=(const ShaderEntity&) -> ShaderEntity& = default;
   auto operator=(ShaderEntity&&) -> ShaderEntity& = delete;

   // hmm
   virtual ~ShaderEntity() = 0;

private:
   GLuint shaderID;
};

inline auto ShaderEntity::displaySetupErrors() const -> void {
   GLint buildStatus = glGetSetupiv();
   if (GL_TRUE == buildStatus) {
      return;
   }

   std::string infoLog = glGetInfoLog();
   std::cerr << "Shader Setup Status: " << buildStatus << " (==" TO_STR(GL_TRUE)
      << "==" X_TO_STR(GL_TRUE) " expected) "

      << "while compiling/linking: " << typeid(this).name() << "\n"
      << infoLog << "\n===\n";
}

// TODO: rename me to shader program??
class Shader {
   public:
    // use/activate the shader program
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
