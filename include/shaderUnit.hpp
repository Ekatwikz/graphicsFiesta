#ifndef SHADER_UNIT_HPP
#define SHADER_UNIT_HPP

#include "shaderEntity.hpp"
#include "file.hpp"

// TODO: Type variant thingy?

template <GLenum SHADER_TYPE>
class ShaderUnit : public ShaderEntity {
   public:
    auto setup() const -> void override;

    // not sure if this is okay tbh
    explicit ShaderUnit(File&& shaderFile_);

    // base class stuffs
    [[nodiscard]] auto glGetSetupiv() const -> GLint override;
    [[nodiscard]] auto glGetInfoLog() const -> std::string override;
    [[nodiscard]] auto getName() const -> std::string override {
        return typeid(this).name();
    };

    // only move-constructble?
    ShaderUnit(const ShaderUnit&) = delete;
    ShaderUnit(ShaderUnit&&) noexcept = default;
    auto operator=(const ShaderUnit&) -> ShaderUnit& = delete;
    auto operator=(ShaderUnit&&) -> ShaderUnit& = delete;

    ShaderUnit() = default;
    ~ShaderUnit() override { glDeleteShader(getID()); }

   private:
    File shaderFile;
};

template <GLenum SHADER_TYPE>
inline auto ShaderUnit<SHADER_TYPE>::setup() const -> void {
    glShaderSource(getID(), 1, shaderFile.getRawContentP(), nullptr);
    glCompileShader(getID());
}

template <GLenum SHADER_TYPE>
inline ShaderUnit<SHADER_TYPE>::ShaderUnit(File&& shaderFile_)
    : shaderFile{shaderFile_} {
    setID(glCreateShader(SHADER_TYPE));
    setup();
}

template <GLenum SHADER_TYPE>
inline auto ShaderUnit<SHADER_TYPE>::glGetSetupiv() const -> GLint {
    GLint compileStatus = 0;
    glGetShaderiv(getID(), GL_COMPILE_STATUS, &compileStatus);
    return compileStatus;
}

template <GLenum SHADER_TYPE>
inline auto ShaderUnit<SHADER_TYPE>::glGetInfoLog() const -> std::string {
    // TODO: get the string size in a smarter way,
    // probably isn't very smart to do it like this
    std::string infoLog(1024, '\0');

    glGetShaderInfoLog(getID(), static_cast<GLsizei>(infoLog.size()), nullptr,
                        infoLog.data());
    return infoLog;
}

#endif // SHADER_UNIT_HPP
