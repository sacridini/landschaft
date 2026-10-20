// SPDX-License-Identifier: GPL-3.0-or-later

#include "swat/fio.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace swat::fio {

namespace {

bool is_blank(char c) { return c == ' ' || c == '\t' || c == '\r'; }

#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
#define SWAT_FIO_FROM_CHARS 1
#endif

// A real of the whole of `t` (normalize_real's form), as std::from_chars reads it. Apple's libc++ has no
// from_chars of reals: strtof/strtod there (the "C" locale), as correctly rounded.
template <class T>
std::errc parse_real(const std::string& t, T& v) {
#if defined(SWAT_FIO_FROM_CHARS)
    const auto r = std::from_chars(t.data(), t.data() + t.size(), v, std::chars_format::general);
    if (r.ec != std::errc()) return r.ec;
    return r.ptr == t.data() + t.size() ? std::errc() : std::errc::invalid_argument;
#else
    if (t.empty() || std::isspace(static_cast<unsigned char>(t[0]))) return std::errc::invalid_argument;
    char* end = nullptr;
    errno = 0;
    T x;
    if constexpr (std::is_same_v<T, float>) {
        x = std::strtof(t.c_str(), &end);
    } else {
        x = std::strtod(t.c_str(), &end);
    }
    if (end != t.c_str() + t.size()) return std::errc::invalid_argument;
    // ERANGE of a subnormal too, which from_chars reads
    if (errno == ERANGE && (x == 0 || std::isinf(x))) return std::errc::result_out_of_range;
    v = x;
    return std::errc();
#endif
}

// Fortran real text in the form std::from_chars reads: no '+', 'd'/'q' exponents as
// 'e', an exponent sign without a letter ("1.5+3") given its 'e'.
bool normalize_real(std::string_view s, std::string& out) {
    out.clear();
    if (s.empty()) return false;
    std::size_t i = 0;
    if (s[0] == '+' || s[0] == '-') {
        if (s[0] == '-') out.push_back('-');
        i = 1;
    }
    if (i >= s.size()) return false;
    {
        std::string low;
        for (std::size_t k = i; k < s.size(); ++k) low.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(s[k]))));
        if (low == "inf" || low == "infinity" || low == "nan" || low.rfind("nan(", 0) == 0) {
            out += low.rfind("nan", 0) == 0 ? "nan" : "inf";
            return true;
        }
    }
    bool digits = false, dot = false, exp = false;
    for (; i < s.size(); ++i) {
        const char c = s[i];
        if (c >= '0' && c <= '9') {
            out.push_back(c);
            digits = true;
        } else if (c == '.' && !dot && !exp) {
            out.push_back('.');
            dot = true;
        } else if (!exp && digits && (c == 'e' || c == 'E' || c == 'd' || c == 'D' || c == 'q' || c == 'Q')) {
            out.push_back('e');
            exp = true;
            if (i + 1 < s.size() && (s[i + 1] == '+' || s[i + 1] == '-')) {
                if (s[i + 1] == '-') out.push_back('-');
                ++i;
            }
            if (i + 1 >= s.size()) return false;
        } else if (!exp && digits && (c == '+' || c == '-')) {
            out.push_back('e');
            if (c == '-') out.push_back('-');
            exp = true;
            if (i + 1 >= s.size()) return false;
        } else {
            return false;
        }
    }
    return digits;
}

}  // namespace

bool to_float(std::string_view s, float& out) {
    std::string t;
    if (!normalize_real(s, t)) return false;
    if (t == "inf" || t == "-inf") {
        out = t[0] == '-' ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        return true;
    }
    if (t == "nan" || t == "-nan") {
        out = std::numeric_limits<float>::quiet_NaN();
        return true;
    }
    float v = 0.0f;
    const auto ec = parse_real(t, v);
    if (ec == std::errc::result_out_of_range) {
        // overflow to infinity, underflow to zero, as the runtime rounds
        double d = 0.0;
        parse_real(t, d);
        v = static_cast<float>(d);
    } else if (ec != std::errc()) {
        return false;
    }
    out = v;
    return true;
}

bool to_double(std::string_view s, double& out) {
    std::string t;
    if (!normalize_real(s, t)) return false;
    if (t == "inf" || t == "-inf") {
        out = t[0] == '-' ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
        return true;
    }
    if (t == "nan" || t == "-nan") {
        out = std::numeric_limits<double>::quiet_NaN();
        return true;
    }
    double v = 0.0;
    if (parse_real(t, v) != std::errc()) return false;
    out = v;
    return true;
}

bool to_int(std::string_view s, long long& out) {
    if (s.empty()) return false;
    std::size_t i = 0;
    bool neg = false;
    if (s[0] == '+' || s[0] == '-') {
        neg = s[0] == '-';
        i = 1;
    }
    if (i >= s.size()) return false;
    long long v = 0;
    const auto r = std::from_chars(s.data() + i, s.data() + s.size(), v);
    if (r.ec != std::errc() || r.ptr != s.data() + s.size()) return false;
    out = neg ? -v : v;
    return true;
}

File::File(std::shared_ptr<const std::string> text) : text_(std::move(text)) {
    const std::string& t = *text_;
    std::size_t a = 0;
    while (a < t.size()) {
        std::size_t b = t.find('\n', a);
        if (b == std::string::npos) b = t.size();
        std::size_t e = b;
        if (e > a && t[e - 1] == '\r') --e;
        recs_.emplace_back(t.data() + a, e - a);
        a = b + 1;
    }
}

int File::read_line(std::string& line) {
    if (pos_ >= recs_.size()) return -1;
    line.assign(recs_[pos_]);
    ++pos_;
    return 0;
}

bool File::Statement::start_record() {
    if (f_.pos_ >= f_.recs_.size()) return false;
    rec_ = f_.recs_[f_.pos_++];
    col_ = 0;
    started_ = true;
    return true;
}

File::Statement::Tok File::Statement::next() {
    if (repeat_left_ > 0) {
        --repeat_left_;
        if (repeat_null_) return Tok::null;
        tok_ = repeat_tok_;
        return Tok::value;
    }
    if (done_) return iostat_ < 0 ? Tok::end : (iostat_ > 0 ? Tok::error : Tok::slash);
    if (!started_ && !start_record()) {
        done_ = true;
        iostat_ = -1;
        return Tok::end;
    }
    // skip blanks, going on to the next records at their ends
    for (;;) {
        while (col_ < rec_.size() && is_blank(rec_[col_])) ++col_;
        if (col_ < rec_.size()) break;
        if (!start_record()) {
            done_ = true;
            iostat_ = -1;
            return Tok::end;
        }
    }
    const char c = rec_[col_];
    if (c == ',') {
        ++col_;
        return Tok::null;
    }
    if (c == '/') {
        done_ = true;
        return Tok::slash;
    }
    tok_.clear();
    if (c == '\'' || c == '"') {
        ++col_;
        for (;;) {
            if (col_ >= rec_.size()) {
                // a quoted string goes on in the next record
                if (!start_record()) break;
                continue;
            }
            const char d = rec_[col_++];
            if (d == c) {
                if (col_ < rec_.size() && rec_[col_] == c) {
                    tok_.push_back(c);
                    ++col_;
                    continue;
                }
                break;
            }
            tok_.push_back(d);
        }
    } else {
        while (col_ < rec_.size() && !is_blank(rec_[col_]) && rec_[col_] != ',' && rec_[col_] != '/') tok_.push_back(rec_[col_++]);
        // r*c and r*
        const std::size_t star = tok_.find('*');
        if (star != std::string::npos && star > 0 &&
            std::all_of(tok_.begin(), tok_.begin() + static_cast<std::ptrdiff_t>(star), [](char ch) { return ch >= '0' && ch <= '9'; })) {
            const int r = std::stoi(tok_.substr(0, star));
            repeat_tok_ = tok_.substr(star + 1);
            repeat_null_ = repeat_tok_.empty();
            repeat_left_ = r - 1;
            if (repeat_null_) {
                // consume the separator
                while (col_ < rec_.size() && is_blank(rec_[col_])) ++col_;
                if (col_ < rec_.size() && rec_[col_] == ',') ++col_;
                return Tok::null;
            }
            tok_ = repeat_tok_;
        }
    }
    // consume the separator after the value
    while (col_ < rec_.size() && is_blank(rec_[col_])) ++col_;
    if (col_ < rec_.size() && rec_[col_] == ',') ++col_;
    return Tok::value;
}

void File::Statement::item(int& v) {
    const Tok t = next();
    if (t != Tok::value) return;
    long long x = 0;
    if (!to_int(tok_, x) || x < std::numeric_limits<int>::min() || x > std::numeric_limits<int>::max()) {
        done_ = true;
        iostat_ = 1;
        repeat_left_ = 0;
        return;
    }
    v = static_cast<int>(x);
}

void File::Statement::item(long long& v) {
    const Tok t = next();
    if (t != Tok::value) return;
    long long x = 0;
    if (!to_int(tok_, x)) {
        done_ = true;
        iostat_ = 1;
        repeat_left_ = 0;
        return;
    }
    v = x;
}

void File::Statement::item(float& v) {
    const Tok t = next();
    if (t != Tok::value) return;
    float x = 0.0f;
    if (!to_float(tok_, x)) {
        done_ = true;
        iostat_ = 1;
        repeat_left_ = 0;
        return;
    }
    v = x;
}

void File::Statement::item(double& v) {
    const Tok t = next();
    if (t != Tok::value) return;
    double x = 0.0;
    if (!to_double(tok_, x)) {
        done_ = true;
        iostat_ = 1;
        repeat_left_ = 0;
        return;
    }
    v = x;
}

void File::Statement::item(bool& v) {
    const Tok t = next();
    if (t != Tok::value) return;
    std::size_t i = (!tok_.empty() && tok_[0] == '.') ? 1 : 0;
    const char c = i < tok_.size() ? static_cast<char>(std::tolower(static_cast<unsigned char>(tok_[i]))) : '\0';
    if (c == 't') {
        v = true;
    } else if (c == 'f') {
        v = false;
    } else {
        done_ = true;
        iostat_ = 1;
        repeat_left_ = 0;
    }
}

void File::Statement::item(Chars v) {
    const Tok t = next();
    if (t != Tok::value) return;
    v.s = v.len < tok_.size() ? tok_.substr(0, v.len) : tok_;
    // Fortran keeps trailing blanks as padding only: compare without them
    while (!v.s.empty() && v.s.back() == ' ') v.s.pop_back();
}

int File::Statement::finish() {
    // a READ with no items, or whose items were all given by repeats, still takes a record
    if (!started_ && !done_ && !start_record()) return -1;
    return iostat_;
}

void FileSet::add(const std::string& name, std::string text) {
    files_[key(name)] = {name, std::make_shared<const std::string>(std::move(text))};
}

bool FileSet::exists(const std::string& name) const { return files_.count(key(name)) > 0; }

File FileSet::open(const std::string& name) const {
    const auto it = files_.find(key(name));
    if (it == files_.end()) return File();
    return File(it->second.second);
}

std::vector<std::string> FileSet::names() const {
    std::vector<std::string> out;
    for (const auto& [k, v] : files_) out.push_back(v.first);
    return out;
}

std::string FileSet::key(const std::string& name) {
    std::string k;
    std::size_t a = 0, b = name.size();
    while (a < b && name[a] == ' ') ++a;
    while (b > a && name[b - 1] == ' ') --b;
    for (std::size_t i = a; i < b; ++i) {
        char c = name[i];
        if (c == '\\') c = '/';
        k.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    while (k.rfind("./", 0) == 0) k.erase(0, 2);
    return k;
}

}  // namespace swat::fio
