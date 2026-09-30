// SPDX-License-Identifier: GPL-3.0-or-later
//
// Spatial prioritization after Zonation 5 v2.4 (Lehtinen, Moilanen et al.,
// University of Helsinki; GPL-3): feature layers quantized to 8 or 16 bits,
// ranking by iterated stable sorts of the cells by their marginal loss at the
// point of their removal. Same results as the z5 binary on one thread; see
// IDEAS.md "Priorização espacial sistemática (Zonation 5)".
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace prio {

using i64 = std::int64_t;

// How the values of the feature layers are read (Zonation's value extraction,
// NoData and thresholds).
struct ReadRules {
    int layers = 0;
    std::vector<float> nodata;      // the band's NoData value (NaN: none; NaN cells are always NoData)
    std::vector<float> extract;     // value extraction ("file.tif/3"), NaN: none
    std::vector<float> thresholds;  // values below become 0 (0: off)
};

// Statistics of the layers (Zonation's MapStats), gathered block of rows by block.
struct Stats {
    explicit Stats(ReadRules rules);
    // values: (layers, h, w) C order; area: (h, w), 0 = outside the analysis area (or null)
    void add_block(const float* values, i64 h, i64 w, const std::uint8_t* area);

    ReadRules rules;
    i64 pixels = 0, data_cells = 0;
    std::vector<i64> zeros, negative, nodata;
    std::vector<float> min_values, max_values, orig_max_values;
};

// The quantized feature data (Zonation's LayerData), built block of rows by block
// from the layers that are not empty.
struct Data {
    Data(ReadRules rules, std::vector<float> max_values, std::vector<std::uint8_t> keep_zeros, int bits);
    // cond: (conditions, h, w) condition values in [0, 1] (Zonation's -c), or null
    void add_block(const float* values, i64 h, i64 w, const std::uint8_t* area, const float* cond = nullptr);
    void flush();   // end of one of Zonation's blocks of rows: add its sums
    void finish();
    // Zonation's retention transform of a layer (-r): kind 0 absolute values, 1
    // fractions (per data cell), 2 a constant fraction; returns the cells where the
    // absolute retention exceeded the value (clipped)
    i64 retain(int layer, int kind, const float* values, float fraction);
    // Zonation's pre-processing maps (-b): per data cell, the sum over the layers of
    // mult * (weight *) value, as sum-normalized values or scaled back to the input sums
    std::vector<float> preprocessing(const float* mult, const float* weights, bool unscaled) const;

    std::vector<int> condition;             // condition of each layer (-1: none)
    std::vector<std::uint8_t> renormalize;  // renormalize a conditioned layer to level 1

    ReadRules rules;
    int bits = 8;
    double storage_scale = 255.0;  // 2^bits - 1
    std::vector<float> max_values, inv_max;
    std::vector<std::uint8_t> keep_zeros;
    i64 pixels = 0;  // cells seen so far (row-major index of the next one)

    // cells with data, in row-major order: their index in the map and their entries
    std::vector<i64> cell_index;
    std::vector<i64> cell_ptr{0};
    std::vector<std::uint16_t> entry_layer;
    std::vector<std::uint16_t> entry_value;  // quantized value (0 .. 2^bits - 1)

    // per layer (after finish)
    std::vector<double> max_norm_sums, nq_sums, condition_sums, orig_abs_sums;
    std::vector<double> layer_scale, initial_levels, quantization_errors;
    std::vector<i64> quantized_zeros;

    i64 cells() const { return static_cast<i64>(cell_index.size()); }

  private:
    std::vector<double> block_sums_, block_nq_, block_cond_;
};

enum class Rule { ABF, CAZ1, CAZ2, CAZMAX, CAZP, CAZPW, RAND, LOAD };

struct RankSettings {
    Rule rule = Rule::CAZ2;
    double param = 0.0;              // ABF z, CAZP/CAZPW p (Zonation's float, as double)
    std::vector<float> weights;      // per layer (1 when weights are off)
    std::vector<float> abf_z;        // per layer, NaN: param
    std::vector<float> cost;         // per data cell, empty: none
    bool cost_in_ranking = false;
    std::vector<std::uint8_t> hmask; // per data cell, empty: none
    std::vector<float> solution;     // per data cell (LOAD)
    int max_iterations = 500;
    float ranking_error = 0.01f;     // percent
    bool stop_on_plateau = false;
    float smoothing = 0.0f;
    float curves_interval = 0.0001f;
    std::uint64_t seed = 0;          // RAND
};

struct RankResult {
    std::vector<float> rank;        // per data cell
    std::vector<std::uint32_t> order;  // data cells in removal order
    int iterations = 0;
    double ranking_error = 0.0;     // fraction (Zonation prints it * 100)
    i64 ranking_errors = 0;
    double rank_change = 0.0;
    std::vector<double> history_error, history_change;
    std::vector<i64> history_errors;
    // curves of the returned ranking
    std::vector<float> curve_rank;        // samples
    std::vector<float> curve_levels;      // (samples, layers)
    std::vector<double> curve_cost;       // samples
    double auc = 0.0;
    double total_cost = 0.0;
};

RankResult rank(const Data& data, const RankSettings& settings);

// Zonation's sum of a float column (Eigen 3.4's reduction with AVX packets of 8,
// the column starting `offset` floats after a 32-byte boundary).
float eigen_avx_sum(const float* v, i64 n, i64 offset);

}  // namespace prio
