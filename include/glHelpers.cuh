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
                 break; \
                 glErrorName = TO_STR(GL_TABLE_TOO_LARGE); */ \
            default: \
                glErrorName = "???"; \
                break; \
        } \
        fprintf(stderr, __FILE__ ":%d in %s | glGetError()->0x%08X (%s)\n", __LINE__, static_cast<const char*>(__func__), glErrorVal, glErrorName); \
    } \
    glErrorVal; \
}))

#endif // GL_HELPERS_CUH
