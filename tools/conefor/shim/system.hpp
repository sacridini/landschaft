// Stand-in for Borland's <system.hpp>: only what landscape.h needs.
#pragma once
#include <string>
#include <type_traits>
struct AnsiString : std::string {
    AnsiString() = default;
    AnsiString(const char* s) : std::string(s) {}
    AnsiString(const std::string& s) : std::string(s) {}
    template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
    AnsiString(T v) : std::string(std::to_string(v)) {}
};
