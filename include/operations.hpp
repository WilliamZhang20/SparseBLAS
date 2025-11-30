#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <ranges>
#include <span>
#include "sparse_vector.hpp"
#include "csr_matrix.hpp"

namespace slinalg {

// dense opeartions (operate on std::span in contiguous memory)

template <class T>
inline T dot(std::span<const T&> a, std::span<const T&> b) {
    return std::transform_reduce(
        a.begin(), a.end(), b.begin(),
        0., std::plus<>{}, std::multiplies<>{});
};

template <class T>
inline double norm2(std::span<const T&> v) {
    return std::sqrt(std::transform_reduce(
        v.begin(), v.end(), 0.0,
        std::plus<>{},
        [](double x){ return x * x; }
    ));
}

inline void axpy(double alpha, std::span<const double> x, std::span<double> y) {
    for (size_t i = 0; i < x.size(); ++i)
        y[i] += alpha * x[i];
}

// sparse operations

// Sparse matrix-vector multiply: y = A*x
template <class T>
inline void spmv(const CSRMatrix<T>& A,
            const std::vector<T>& x,
            std::vector<T>& y)
{
    assert(int(x.size()) == A.ncols);
    y.assign(A.nrows, T{});

    for(int r=0; r < A.nrows; ++r) {
        T acc = 0;
        for(auto [c, v] : A.row(r)) {
            acc += v * x[c];
        }
        y[r] = acc;
    }
}


template <class T>
inline T sDot(const SparseVector<T>& a, const SparseVector<T>& b) {
    T sum = 0;

    auto ia = a.indices.begin();
    auto ib = b.indices.begin();
    auto va = a.values.begin();
    auto vb = b.values.begin();

    // skip unnecessary multiplications for sparsity
    while(ia != a.indices.end() && ib != b.indices.end()) {
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

}