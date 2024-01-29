#ifndef SHADER_PROGRAM_HPP
#define SHADER_PROGRAM_HPP

#include <algorithm>

#include <variant>
#include <vector>

#include "shaderEntity.hpp"
#include "shaderUnit.hpp"

class ShaderProgram : ShaderEntity {
   public:
    // use/activate the shader program
    auto glUseProgram() const -> void;

    // glUniform setter,
    // but like... should this really be const??
    template <typename T>
    auto glUniform(const GLchar* name, T value) const -> void;

    // base class stuffs
    [[nodiscard]] auto glGetSetupiv() const -> GLint override;
    [[nodiscard]] auto glGetInfoLog() const -> std::string override;
    [[nodiscard]] auto getName() const -> std::string override {
        return typeid(this).name();
    };

   auto setup() const -> void override { glLinkProgram(getID()); };

   // TODO: make the type better here lol, ugh
   template<typename... Args>
   explicit ShaderProgram(Args&&... args);

    // no move, no copy, xdd
    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram(ShaderProgram&&) = delete;
    auto operator=(const ShaderProgram&) -> ShaderProgram& = delete;
    auto operator=(ShaderProgram&&) -> ShaderProgram& = delete;

    ~ShaderProgram() override { glDeleteProgram(getID()); }

private:
   // the commented out ones are officially supported,
   // but aren't in the glad loader headers for whatever reason lol
   using ShaderUnitVariant = std::variant<
   //ShaderUnit<GL_COMPUTE_SHADER>,
   ShaderUnit<GL_VERTEX_SHADER>,
   //ShaderUnit<GL_TESS_CONTROL_SHADER>,
   //ShaderUnit<GL_TESS_EVALUATION_SHADER>,
   ShaderUnit<GL_GEOMETRY_SHADER>,
   ShaderUnit<GL_FRAGMENT_SHADER>
   >;

   std::vector<ShaderUnitVariant> shaderUnits;
};

template<typename... Args>
inline ShaderProgram::ShaderProgram(Args&&... args) {
   // https://en.cppreference.com/w/cpp/language/sizeof...
   shaderUnits.reserve(sizeof...(args));

   // C++ is a functional lang now? 💀💀
   // we out here foldin
   (shaderUnits.emplace_back(std::forward<Args>(args)), ...);

   for (const auto& shaderUnit : shaderUnits) {
      std::visit([&](auto&& shader) {
         shader.displaySetupErrors();
         glAttachShader(getID(), shader.getID());
      }, shaderUnit);
   }

   setup();
   displaySetupErrors();
}

#endif // SHADER_PROGRAM_HPP
