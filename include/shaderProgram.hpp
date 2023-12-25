#ifndef SHADER_H
#define SHADER_H

#include "shaderEntity.hpp"

class ShaderProgram : ShaderEntity {
   public:
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

    ~ShaderProgram() override;
};

// TODO: DELETE ME!!
enum class ShaderType { PROGRAM, VERTEX, FRAGMENT };
static inline auto operator<<(std::ostream& outputStream,
                              ShaderType shaderTypeEnum) -> std::ostream&;

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

#endif  // SHADER_H
