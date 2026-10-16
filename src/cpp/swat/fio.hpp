// SPDX-License-Identifier: GPL-3.0-or-later
//
// Fortran list-directed input (READ (unit, *)) over in-memory text files, as SWAT+
// reads its TxtInOut: each READ starts at the next record and goes on to later
// records while it needs values, the rest of its last record is skipped; values are
// separated by blanks, tabs, commas or the end of a record; a slash ends the
// statement; two commas give a null value (the item keeps its value); "r*c" repeats.
// Text becomes float or double by correct rounding, as the Fortran runtime does (never
// through a double for a float). iostat: 0, -1 at the end of the file, 1 on a bad value
// (the items before it keep what they read, the later ones their old values).

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace swat::fio {

// A character item of a given declared length: longer values are cut, as in Fortran.
struct Chars {
    std::string& s;
    std::size_t len;
};
inline Chars chars(std::string& s, std::size_t len) { return {s, len}; }

// n consecutive items (an array section such as x(1:12)).
template <class T>
struct Items {
    T* p;
    int n;
};
template <class T>
Items<T> items(T* p, int n) {
    return {p, n};
}

// Type-erased items: the components of a derived type, listed for one READ.
struct Item {
    enum Kind { I32, I64, F32, F64, BOOL, CHARS } kind;
    void* p;
    std::size_t len;
};
using ItemList = std::vector<Item>;
inline void add(ItemList& l, int& v) { l.push_back({Item::I32, &v, 0}); }
inline void add(ItemList& l, long long& v) { l.push_back({Item::I64, &v, 0}); }
inline void add(ItemList& l, float& v) { l.push_back({Item::F32, &v, 0}); }
inline void add(ItemList& l, double& v) { l.push_back({Item::F64, &v, 0}); }
inline void add(ItemList& l, bool& v) { l.push_back({Item::BOOL, &v, 0}); }
inline void add(ItemList& l, Chars c) { l.push_back({Item::CHARS, &c.s, c.len}); }
inline void add(ItemList& l, const ItemList& m) { l.insert(l.end(), m.begin(), m.end()); }
template <class T>
void add(ItemList& l, Items<T> v) {
    for (int i = 0; i < v.n; ++i) add(l, v.p[i]);
}

class File {
public:
    File() = default;
    explicit File(std::shared_ptr<const std::string> text);

    bool open() const { return text_ != nullptr; }
    void rewind() { pos_ = 0; }
    void backspace() {
        if (pos_ > 0) --pos_;
    }
    std::size_t records() const { return recs_.size(); }

    // READ (unit, *) items...; returns iostat.
    template <class... T>
    int read(T&&... items) {
        Statement st(*this);
        (st.item(items), ...);
        return st.finish();
    }

    // READ (unit, '(A)') line: the whole next record.
    int read_line(std::string& line);

    class Statement;
    using Reader = Statement;
    Statement reader();

    class Statement {
    public:
        explicit Statement(File& f) : f_(f) {}
        void item(int& v);
        void item(long long& v);
        void item(float& v);
        void item(double& v);
        void item(bool& v);
        void item(std::string& v) { item(Chars{v, std::string::npos}); }
        void item(Chars v);
        template <class T>
        void item(Items<T> v) {
            for (int i = 0; i < v.n; ++i) item(v.p[i]);
        }
        void item(const ItemList& l) {
            for (const Item& it : l) {
                switch (it.kind) {
                    case Item::I32: item(*static_cast<int*>(it.p)); break;
                    case Item::I64: item(*static_cast<long long*>(it.p)); break;
                    case Item::F32: item(*static_cast<float*>(it.p)); break;
                    case Item::F64: item(*static_cast<double*>(it.p)); break;
                    case Item::BOOL: item(*static_cast<bool*>(it.p)); break;
                    case Item::CHARS: item(Chars{*static_cast<std::string*>(it.p), it.len}); break;
                }
            }
        }
        int finish();

    private:
        enum class Tok { value, null, end, slash, error };
        // Next value of the statement: its text in tok_, or why there is none.
        Tok next();
        bool start_record();
        File& f_;
        bool started_ = false;
        bool done_ = false;  // slash, end of file or error seen: later items keep their values
        int iostat_ = 0;
        std::string_view rec_;
        std::size_t col_ = 0;
        std::string tok_;
        int repeat_left_ = 0;
        bool repeat_null_ = false;
        std::string repeat_tok_;
        bool after_value_ = false;  // a value was read in this statement (a comma then separates)
    };

private:
    std::shared_ptr<const std::string> text_;
    std::vector<std::string_view> recs_;
    std::size_t pos_ = 0;
};

inline File::Statement File::reader() { return Statement(*this); }

// The files of a project, looked up by name without regard to case (the reference
// runs on Windows, where "Ames.pcp" opens "ames.pcp").
class FileSet {
public:
    void add(const std::string& name, std::string text);
    bool exists(const std::string& name) const;
    File open(const std::string& name) const;  // a closed File when missing
    std::vector<std::string> names() const;

private:
    static std::string key(const std::string& name);
    std::map<std::string, std::pair<std::string, std::shared_ptr<const std::string>>> files_;
};

// Correctly rounded conversions of Fortran real text ("1.5", ".5", "5.", "1e3", "1.d3",
// "-2.5E-01", "1+3"); false when the text is not a number.
bool to_float(std::string_view s, float& out);
bool to_double(std::string_view s, double& out);
bool to_int(std::string_view s, long long& out);

}  // namespace swat::fio
