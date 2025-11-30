#pragma once
#include <vector>
#include <ranges>
#include <cassert>

template <class T>
class CSCMatrix {
public:
    int nrows = 0;
    int ncols = 0;

    std::vector<int> colptr;  // starting position of each column
    std::vector<int> rowind;  // row index per value
    std::vector<T>   values;  // nnz values

    CSCMatrix() = default;

    CSCMatrix(int r, int c,
              std::vector<int> cp,
              std::vector<int> ri,
              std::vector<T>   val)
        : nrows(r), ncols(c),
          colptr(std::move(cp)),
          rowind(std::move(ri)),
          values(std::move(val))
    {
        assert(colptr.size() == size_t(ncols + 1));
        assert(rowind.size() == values.size());
    }

    int nnz() const noexcept { return values.size(); }

    // return zipped (row, val) for column c
    auto col(int c) const {
        assert(c >= 0 && c < ncols);
        int start = colptr[c];
        int count = colptr[c+1] - start;

        auto row_slice = rowind | std::views::drop(start) | std::views::take(count);
        auto val_slice = values | std::views::drop(start) | std::views::take(count);
        return std::views::zip(row_slice, val_slice);
    }
};
