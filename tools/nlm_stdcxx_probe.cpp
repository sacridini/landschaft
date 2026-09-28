// SPDX-License-Identifier: GPL-3.0-or-later
//
// Draws from std::mt19937 with the distributions of the C++ library it is
// compiled with: built with the Rtools g++ (the compiler that builds NLMR's
// C++ code) by tools/make_reference_nlm.py to check landschaft's emulation
// of libstdc++ (src/cpp/nlm/rng.cpp).
//
// Usage: nlm_stdcxx_probe <commands> <output>
// Commands, one per line: "seed S" (a new std::mt19937(S)), "canonical N",
// "uniform_int N A B", "normal N MEAN SD" (one distribution object),
// "normal_fresh N MEAN SD" (a new distribution per draw), "shuffle N"
// (the permutation of 0..N-1). Every command after "seed" writes N doubles
// to the binary output.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: nlm_stdcxx_probe <commands> <output>\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    std::ofstream out(argv[2], std::ios::binary);
    std::mt19937 mt;
    std::string line;
    auto put = [&](double v) { out.write(reinterpret_cast<const char*>(&v), sizeof v); };
    while (std::getline(in, line)) {
        std::istringstream s(line);
        std::string kind;
        if (!(s >> kind)) continue;
        if (kind == "seed") {
            unsigned long seed;
            s >> seed;
            mt.seed(seed);
            continue;
        }
        long long n;
        s >> n;
        if (kind == "canonical") {
            std::uniform_real_distribution<double> d(0, 1);
            for (long long i = 0; i < n; ++i) put(d(mt));
        } else if (kind == "uniform_int") {
            long long a, b;
            s >> a >> b;
            std::uniform_int_distribution<int> d(static_cast<int>(a), static_cast<int>(b));
            for (long long i = 0; i < n; ++i) put(d(mt));
        } else if (kind == "normal" || kind == "normal_fresh") {
            double mean, sd;
            s >> mean >> sd;
            std::normal_distribution<double> d(mean, sd);
            for (long long i = 0; i < n; ++i) {
                if (kind == "normal_fresh") d = std::normal_distribution<double>(mean, sd);
                put(d(mt));
            }
        } else if (kind == "shuffle") {
            std::vector<long long> v(static_cast<std::size_t>(n));
            for (long long i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = i;
            std::shuffle(v.begin(), v.end(), mt);
            for (long long x : v) put(static_cast<double>(x));
        } else {
            std::fprintf(stderr, "unknown command: %s\n", kind.c_str());
            return 1;
        }
    }
    return 0;
}
