#pragma once

#include <sstream>
#include <string>

namespace ml {

class RuntimeException {
public:
    explicit RuntimeException(const std::string& what)
        : what_(what), line_(0) {}

    template <typename T>
    friend RuntimeException& operator << (RuntimeException& e, const T& t) {
        std::ostringstream oss;
        oss << t;
        e.what_ += oss.str();
        return e;
    }

    template <typename T>
    friend RuntimeException&& operator << (RuntimeException&& e, const T& t) {
        std::ostringstream oss;
        oss << t;
        e.what_ += oss.str();
        return std::move(e);
    }

    RuntimeException& line(unsigned l) {
        line_ = l;
        return *this;
    }

    unsigned line() const {
        return line_;
    }

    RuntimeException& file(std::string f) {
        file_ = std::move(f);
        return *this;
    }

    const std::string& file() const {
        return file_;
    }

    const std::string& what() const {
        return what_;
    }

private:
    std::string what_;
    unsigned line_;
    std::string file_;
};

}

#define MAKE_EX(Exc, what) \
    Exc{what}.line(__LINE__).file(__FILE__)

#ifdef NDEBUG
#define ASSERT(expr) ((void)0)
#else
#define ASSERT(expr) \
    if (!(expr)) { \
        throw MAKE_EX(ml::RuntimeException, "Assertion failed: " #expr); \
    }
#endif

#ifdef NDEBUG
#define REQUIRE(expr, msg) ((void)0)
#else
#define REQUIRE(expr, msg) \
    if (!(expr)) { \
        throw MAKE_EX(ml::RuntimeException, \
                      "Requirement is not met: " #expr ".") << msg; \
    }
#endif

