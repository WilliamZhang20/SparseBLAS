#include <vector>
#include <limits>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <iostream>

#include "../include/csr_matrix.hpp"

// -----------------------------------------------------------------------------
// C = A * B   (CSR × CSR → CSR)
// Classical Gustavson-style method.
// -----------------------------------------------------------------------------

template <class T>
CSRMatrix<T> spgemm(const CSRMatrix<T>& A, const CSRMatrix<T>& B)
{
    assert(A.ncols == B.nrows);

    const std::size_t m = A.nrows;
    const std::size_t n = B.ncols;

    std::vector<int> rowptr;
    std::vector<int> colind;
    std::vector<double>      values;

    rowptr.reserve(m + 1);
    rowptr.push_back(0);

    // acc_index[j] = index in colind/values for column j of current row
    // acc_value[j] = value being accumulated
    std::vector<std::size_t> acc_index(n, std::numeric_limits<std::size_t>::max());
    std::vector<double>      acc_value(n, 0.0);
    std::vector<std::size_t> touched; 
    touched.reserve(256);

    for (std::size_t i = 0; i < m; ++i) {
        touched.clear();

        // Traverse row i of A
        for (auto [ak_col, ak_val] : A.row(i)) {

            // Row ak_col of B
            for (auto [bj_col, bj_val] : B.row(ak_col)) {

                if (acc_index[bj_col] == std::numeric_limits<std::size_t>::max()) {
                    // First time touching column bj_col in this row
                    acc_index[bj_col] = colind.size(); // temp index for now
                    acc_value[bj_col] = ak_val * bj_val;
                    touched.push_back(bj_col);
                } else {
                    // Already exists; accumulate
                    acc_value[bj_col] += ak_val * bj_val;
                }
            }
        }

        // Sort touched columns for CSR canonical order
        std::sort(touched.begin(), touched.end());

        // Append to colind/values
        for (std::size_t col : touched) {
            colind.push_back(col);
            values.push_back(acc_value[col]);

            // Reset the accumulator entry for the next row
            acc_index[col] = std::numeric_limits<std::size_t>::max();
        }

        rowptr.push_back(colind.size());
    }

    return CSRMatrix<T>(m, n, rowptr, colind, values);
}

template <typename T>
static void print_csr(const CSRMatrix<T>& M) {
    std::cout << "Matrix " << M.nrows << "x" << M.ncols << "\n";
    for (size_t i = 0; i < M.nrows; ++i) {
        std::cout << " row " << i << ": ";
        for (auto [c, v] : M.row(i)) {
            std::cout << "(" << c << ", " << v << ") ";
        }
        std::cout << "\n";
    }
}

int main() {
    //
    // Test case:
    //
    // A = [ 1 2 0
    //       0 3 4 ]
    //
    // B = [ 5 0
    //       0 6
    //       7 8 ]
    //
    // C = A * B =
    //
    //   = [ 1*5 + 2*0 + 0*7    1*0 + 2*2 + 0*8 ] = [ 5   12 ]
    //     [ 0*5 + 3*0 + 4*7    0*0 + 3*6 + 4*8 ]   [ 28  50 ]
    //

    // ----------------------
    // Build A in CSR
    // ----------------------
    std::vector<int> A_rowptr = {0, 2, 4};
    std::vector<int> A_colind = {0, 1, 1, 2};
    std::vector<double> A_vals   = {1, 2, 3, 4};

    CSRMatrix<double> A(2, 3, A_rowptr, A_colind, A_vals);

    // ----------------------
    // Build B in CSR
    // ----------------------
    std::vector<int> B_rowptr = {0, 1, 2, 4};
    std::vector<int> B_colind = {0, 1, 0, 1};
    std::vector<double> B_vals   = {5, 6, 7, 8};

    CSRMatrix<double> B(3, 2, B_rowptr, B_colind, B_vals);

    // ----------------------
    // Multiply
    // ----------------------
    CSRMatrix<double> C = spgemm(A, B);

    // Print result
    print_csr(C);

    // ----------------------
    // Validate
    // ----------------------
    assert(C.nrows == 2);
    assert(C.ncols == 2);

    // row 0
    {
        auto r = C.row(0);
        std::vector<std::pair<size_t,double>> row0;
        for (auto [c, v] : r) row0.emplace_back(c, v);

        assert(row0.size() == 2);
        assert(row0[0].first == 0 && std::abs(row0[0].second - 5.0) < 1e-12);
        assert(row0[1].first == 1 && std::abs(row0[1].second - 12.0) < 1e-12);
    }

    // row 1
    {
        auto r = C.row(1);
        std::vector<std::pair<size_t,double>> row1;
        for (auto [c, v] : r) row1.emplace_back(c, v);

        assert(row1.size() == 2);
        assert(row1[0].first == 0 && std::abs(row1[0].second - 28.0) < 1e-12);
        assert(row1[1].first == 1 && std::abs(row1[1].second - 50.0) < 1e-12);
    }

    std::cout << "SpGEMM test PASSED.\n";
    return 0;
}
