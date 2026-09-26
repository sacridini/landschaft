// Stand-in for Borland's <SysUtils.hpp>: the Exception class.
#pragma once
#include <stdexcept>
struct Exception : std::runtime_error {
    explicit Exception(const char* m) : std::runtime_error(m) {}
};
