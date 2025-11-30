#pragma once
#include <vector>
#include <ranges>
#include <iostream>
// KEY: nnz = number of non-zero

template <class T>
class SparseVector {
public:
    std::vector<int> indices;   // sorted column indices
    std::vector<T> values;    // non-zero values

    SparseVector() = default;

    SparseVector(std::vector<int> idx, std::vector<T> val)
        : indices(std::move(idx)), values(std::move(val))
    {
        assert(indices.size() == values.size());
    }

    int nnz() const noexcept { return indices.size(); }

    auto entries() const {
        return std::views::zip(indices, values);
    }

    T operator[](int col) const { // returns 0 if not found via binary search
        auto it = std::lower_bound(indices.begin(), indices.end(), col);
        if (it == indices.end() || *it != col) return T{};
        return values[it - indices.begin()];
    }

    void insert(int col, const T& val) {
        auto it = std::lower_bound(indices.begin(), indices.end(), col);
        int pos = it - indices.begin();
        if (it != indices.end() && *it == col) {
            values[pos] = val;
        } else {
            indices.insert(it, col);
            values.insert(values.begin() + pos, val);
        }
    }
};
