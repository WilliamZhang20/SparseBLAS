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

template <class T>
T dot(const SparseVector<T>& a, const SparseVector<T>& b) {
    T sum = 0;

    auto ia = a.indices.begin();
    auto ib = b.indices.begin();
    auto va = a.values.begin();
    auto vb = b.values.begin();

    // skip unnecessary multiplications for sparsity
    while (ia != a.indices.end() && ib != b.indices.end()) {
        if (*ia < *ib) { ++ia; ++va; }
        else if (*ib < *ia) { ++ib; ++vb; }
        else {
            sum += (*va) * (*vb);
            ++ia; ++va;
            ++ib; ++vb;
        }
    }
    return sum;
}