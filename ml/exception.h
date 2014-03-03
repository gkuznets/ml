#pragma once

#include <stdexcept>
#include <string>

namespace ml {

class RuntimeException : public std::runtime_error {
public:
    explicit RuntimeException(const std::string& what)
        : std::runtime_error(what) {}

};

}

#ifdef NDEBUG
#define ASSERT(expr) ((void)0)
#else
#define ASSERT(expr) \
    if (!(expr)) { \
        throw ml::RuntimeException{"Assertion failed: " #expr}; \
    }
#endif

#ifdef NDEBUG
#define REQUIRE(expr, msg) ((void)0)
#else
#define REQUIRE(expr, msg) \
    if (!(expr)) { \
        throw ml::RuntimeException(msg); \
    }
#endif

