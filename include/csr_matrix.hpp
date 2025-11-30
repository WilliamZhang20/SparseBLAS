#pragma once
#include <vector>
#include <ranges>
#include <cassert>
// KEY: nnz = number of non-zero

template <class T>
class CSRMatrix {
public:
    int nrows = 0;
    int ncols = 0;

    std::vector<int> rowptr;
    std::vector<int> colind;
    std::vector<T> values;

    CSRMatrix() = default;

    CSRMatrix(int r, int c,
                std::vector<int> rp,
                std::vector<int> ci,
                std::vector<T> val)
            : nrows(r), ncols(c),
              rowptr(std::move(rp)),
              colind(std::move(ci)),
              values(std::move(val))
    {
        assert(rowptr.size() == size_t(nrows + 1));
        assert(colind.size() == values.size());
    }

    int nnz() const noexcept { return values.size(); }

    // return a zipped range (col, val) for row r
    auto row(int r) const {
        assert(r >= 0 && r < nrows);
        int start = rowptr[r];
        int count = rowptr[r+1] - start;

        auto col_slice = colind | std::views::drop(start) | std::views::take(count);
        auto val_slice = values | std::views::drop(start) | std::views::take(count);
        
        return std::views::zip(col_slice, val_slice);
    }
};

// Sparse matrix-vector multiply: y = A*x
