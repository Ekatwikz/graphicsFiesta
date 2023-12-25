#ifndef SHADER_PROGRAM_HPP
#define SHADER_PROGRAM_HPP

#include "shaderEntity.hpp"

class ShaderProgram : ShaderEntity {
   public:
    auto setup() const -> void override;

    // use/activate the shader program
    auto glUseProgram() const -> void;

    // glUniform setter,
    // but like... should this really be const??
    template <typename T>
    auto glUniform(const GLchar* name, T value) const -> void;

    // constructor reads and builds the shader
    // TODO: don't do it like this lol
    explicit ShaderProgram(const char* vertexShaderPath,
                           const char* fragmentShaderPath);

    // base class stuffs
    [[nodiscard]] auto glGetSetupiv() const -> GLint override;
    [[nodiscard]] auto glGetInfoLog() const -> std::string override;
    [[nodiscard]] auto getName() const -> std::string override {
        return typeid(this).name();
    };

    // no move, no copy, xdd
    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram(ShaderProgram&&) = delete;
    auto operator=(const ShaderProgram&) -> ShaderProgram& = delete;
    auto operator=(ShaderProgram&&) -> ShaderProgram& = delete;

    ~ShaderProgram() override { glDeleteProgram(getID()); }
};

#endif // SHADER_PROGRAM_HPP
