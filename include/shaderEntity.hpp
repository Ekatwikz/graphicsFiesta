#ifndef SHADER_ENTITY
#define SHADER_ENTITY

// include glad to get all the required OpenGL headers
#include <glad/glad.h>
#include <iostream>

#include "tinyHelpers.h"

class ShaderEntity {
public:
   // compiles/links the shader/program
   //virtual auto setup() const -> void = 0;

   // helper function for checking shader compilation/linking errors.
   // TODO: change this to return a string
   auto displaySetupErrors() const -> void;

   // checks if last compile/link was successful
   [[nodiscard]] virtual auto glGetSetupiv() const -> GLint = 0;

   [[nodiscard]] virtual auto glGetInfoLog() const -> std::string = 0;

   [[nodiscard]] auto getID() const { return ID; }

   // for printing errors n stuff ig
   // would be better with some reflection maybe?
   [[nodiscard]] virtual auto getName() const -> std::string = 0;

   // hmm
   ShaderEntity() = default;
   virtual ~ShaderEntity() = 0;

   ShaderEntity(const ShaderEntity&) = default;
   ShaderEntity(ShaderEntity&&) = delete;
   auto operator=(const ShaderEntity&) -> ShaderEntity& = default;
   auto operator=(ShaderEntity&&) -> ShaderEntity& = delete;

protected:
   auto setID(GLuint ID_) { ID = ID_; }

private:
   GLuint ID = 0;
};

// bruhhh what
inline ShaderEntity::~ShaderEntity() = default;

inline auto ShaderEntity::displaySetupErrors() const -> void {
   GLint setupiv = glGetSetupiv();
   if (GL_TRUE == setupiv) {
      return;
   }

   std::string shaderName = getName();
   std::cerr << shaderName << "::" TO_STR(glGetSetupiv) ": -> " << setupiv
      << " (expected: -> " TO_STR(GL_TRUE) " (== " X_TO_STR(GL_TRUE) "))\n"
      << shaderName << "::" TO_STR(glGetInfoLog) ": \""
      << glGetInfoLog() << "\"\n===\n";
}

#endif // !SHADER_ENTITY
