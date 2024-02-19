#ifndef GL_HELPERS_CUH
#define GL_HELPERS_CUH

#define GLCHECK() (__extension__({\
    GLenum glErrorVal; \
    const char* glErrorName; \
    while ((glErrorVal = glGetError()) != GL_NO_ERROR) { \
        switch (glErrorVal) { \
            case GL_INVALID_ENUM: \
                glErrorName = TO_STR(GL_INVALID_ENUM); \
                break; \
            case GL_INVALID_VALUE: \
                glErrorName = TO_STR(GL_INVALID_VALUE); \
                break; \
            case GL_INVALID_OPERATION: \
                glErrorName = TO_STR(GL_INVALID_OPERATION); \
                break; \
            /* case GL_STACK_OVERFLOW: \
                glErrorName = TO_STR(GL_STACK_OVERFLOW); \
                break; \
            case GL_STACK_UNDERFLOW: \
                glErrorName = TO_STR(GL_STACK_UNDERFLOW); \
                break; */ \
            case GL_OUT_OF_MEMORY: \
                glErrorName = TO_STR(GL_OUT_OF_MEMORY); \
                break; \
            case GL_INVALID_FRAMEBUFFER_OPERATION: \
                glErrorName = TO_STR(GL_INVALID_FRAMEBUFFER_OPERATION); \
                break; \
            /* case GL_CONTEXT_LOST: \
                 glErrorName = TO_STR(GL_CONTEXT_LOST); \
                 break; \
             case GL_TABLE_TOO_LARGE: \
                 glErrorName = TO_STR(GL_TABLE_TOO_LARGE); \
                 break; */ \
            default: \
                glErrorName = "???"; \
                break; \
        } \
        fprintf(stderr, __FILE__ ":%d in %s | glGetError()->0x%08X (%s)\n", __LINE__, static_cast<const char*>(__func__), glErrorVal, glErrorName); \
    } \
    glErrorVal; \
}))

// assuming internalFormat and format are the same
#define LOAD_AND_FLIP_TEXTURE_FROM_FILE(filename, format) (__extension__({ \
    int width{}; \
    int height{}; \
    int nrChannels{}; \
\
    stbi_set_flip_vertically_on_load(static_cast<int>(true)); \
\
    uint8_t* data = stbi_load(filename, &width, &height, &nrChannels, 0); \
    if (nullptr != data) { \
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data); \
    } else { \
        std::cerr << "Failed to load texture\n"; \
    } \
\
    stbi_image_free(data); \
}))

#endif // GL_HELPERS_CUH
