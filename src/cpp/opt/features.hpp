// SPDX-License-Identifier: GPL-3.0-or-later
//
// The features x planning units matrix of raster planning units, built from
// blocks of rows of the feature rasters, so that the stack of layers never
// has to be in memory at once. For every layer:
//  - its total (what relative targets are shares of): the sum of its finite
//    values over the whole raster, in cell order, in double (as terra adds
//    them; the blocks keep the order, so the bits do not depend on them);
//  - its amount in every planning unit (cells with a cost): NaN gives no
//    entry, as do zeros; +-inf become +-DBL_MAX (numpy's nan_to_num).
// Layers are independent: they are processed in parallel, with the same bits
// on any number of threads.
#pragma once

#include <cstdint>
#include <vector>

namespace opt {

class FeatureMatrix {
public:
    explicit FeatureMatrix(std::int64_t layers);

    // values: layers x n (row-major), the next n cells; unit: the planning unit of each cell (-1: none)
    void add(const double* values, std::int64_t n, const std::int64_t* unit, int threads);

    std::int64_t layers() const { return static_cast<std::int64_t>(totals_.size()); }
    const std::vector<double>& totals() const { return totals_; }

    // compressed rows (one per layer): start (layers + 1), unit index, amount
    void csr(std::vector<std::int64_t>& start, std::vector<std::int64_t>& index, std::vector<double>& value) const;

private:
    std::vector<double> totals_;
    std::vector<std::vector<std::int64_t>> index_;
    std::vector<std::vector<double>> value_;
};

}  // namespace opt
