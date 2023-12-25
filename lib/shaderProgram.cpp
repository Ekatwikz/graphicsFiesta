#include "shaderProgram.hpp"
#include "shaderUnit.hpp"

// should probably re-setup all its contained ShaderUnits first or something,
// in general...
auto ShaderProgram::setup() const -> void { glLinkProgram(getID()); };
ShaderProgram::ShaderProgram(const char* vertexShaderPath,
                             const char* fragmentShaderPath) {
    ShaderUnit<GL_VERTEX_SHADER> vertexShader{File{vertexShaderPath}};
    vertexShader.displaySetupErrors();

    ShaderUnit<GL_FRAGMENT_SHADER> fragmentShader{File{fragmentShaderPath}};
    fragmentShader.displaySetupErrors();

    // TODO: store the units in some kind of vector or something here
    setID(glCreateProgram());
    glAttachShader(getID(), vertexShader.getID());
    glAttachShader(getID(), fragmentShader.getID());
    setup();
    displaySetupErrors();
}

auto ShaderProgram::glGetSetupiv() const -> GLint {
    GLint linkStatus = 0;
    glGetProgramiv(getID(), GL_LINK_STATUS, &linkStatus);
    return linkStatus;
}

auto ShaderProgram::glGetInfoLog() const -> std::string {
    // TODO: get the string size in a smarter way,
    // probably isn't very smart to do it like this
    std::string infoLog(1024, '\0');

    glGetProgramInfoLog(getID(), static_cast<GLsizei>(infoLog.size()), nullptr,
                        infoLog.data());
    return infoLog;
}

auto ShaderProgram::glUseProgram() const -> void {
    // global namespace operator lul
    ::glUseProgram(getID());
}

template <>
auto ShaderProgram::glUniform<GLint>(const GLchar* name, GLint value) const
    -> void {
    glUniform1i(glGetUniformLocation(getID(), name), value);
}

template <>
auto ShaderProgram::glUniform<GLfloat>(const GLchar* name, GLfloat value) const
    -> void {
    glUniform1f(glGetUniformLocation(getID(), name), value);
}
