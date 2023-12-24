#ifndef TINY_HELPERS_H
#define TINY_HELPERS_H

// useful to force stringify stuffs
// see: https://gcc.gnu.org/onlinedocs/gcc-13.2.0/cpp/Stringizing.html
// see: https://gcc.gnu.org/onlinedocs/gcc-13.2.0/cpp/Argument-Prescan.html
#define TO_STR(s) #s
#define X_TO_STR(X) TO_STR(X)

#endif // TINY_HELPERS_H
