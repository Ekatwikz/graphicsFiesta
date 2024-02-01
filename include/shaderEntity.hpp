#ifndef SHADER_ENTITY_HPP
#define SHADER_ENTITY_HPP

// include glad (if we need to??) to get all the required OpenGL headers
#ifdef __gl_h_
#pragma GCC warning "AIEEEEEE THERES ALREADY OpenGL?!"
#else // __gl_h_
#include <glad/glad.h>
#endif // __gl_h_

#include <iostream>

// print time when something errors out
// could be useful...?
#ifndef NO_DEBUG_TIMESTAMP
#include <chrono>
#include <ctime>
#endif

#include "tinyHelpers.h"

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

    [[nodiscard]] auto getID() const { return ID; }

    // for printing errors n stuff ig
    // would be better with some reflection maybe?
    [[nodiscard]] virtual auto getName() const -> std::string = 0;

    // hmm
    ShaderEntity() = default;
    virtual ~ShaderEntity() = 0;

    // only move construct-ible? idk for now
    ShaderEntity(const ShaderEntity&) = delete;
    ShaderEntity(ShaderEntity&&) = default;
    auto operator=(const ShaderEntity&) -> ShaderEntity& = delete;
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

#ifndef NO_DEBUG_TIMESTAMP
    auto currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cerr << std::ctime(&currentTime);
#endif

    std::string shaderName = getName();
    std::cerr << "Shader ID: " << ID << "\n";
    std::cerr << shaderName << "::" TO_STR(glGetSetupiv) " -> " << setupiv
              << " (expected: -> " TO_STR(GL_TRUE) " (== " X_TO_STR(
                     GL_TRUE) "))\n"
              << shaderName << "::" TO_STR(glGetInfoLog) " -> \""
              << glGetInfoLog() << "\"\n===\n";
}

#endif // SHADER_ENTITY_HPP
