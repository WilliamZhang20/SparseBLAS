#pragma once
#pragma once
#include <vector>
#include <ranges>
#include <cassert>

template <class T>
class COOMatrix {
public:
    int nrows = 0;
    int ncols = 0;

    std::vector<int> rowind;   // row index for each nnz
    std::vector<int> colind;   // col index for each nnz
    std::vector<T>   values;   // nonzero values

    COOMatrix() = default;

    COOMatrix(int r, int c,
              std::vector<int> ri,
              std::vector<int> ci,
              std::vector<T>   val)
        : nrows(r), ncols(c),
          rowind(std::move(ri)),
          colind(std::move(ci)),
          values(std::move(val))
    {
        assert(rowind.size() == colind.size());
        assert(colind.size() == values.size());
    }

    int nnz() const noexcept { return values.size(); }

    // Iterate over all non-zeros as (row, col, val)
    auto entries() const {
        return std::views::zip(rowind, colind, values);
    }

    // Return zipped (col, val) entries for row r
    auto row(int r) const {
        assert(r >= 0 && r < nrows);
        auto zipped = std::views::zip(colind, values);
        return zipped | std::views::filter([&, idx=0](auto const& cv) mutable {
            bool match = (rowind[idx] == r);
            ++idx;
            return match;
        });
    }
};
