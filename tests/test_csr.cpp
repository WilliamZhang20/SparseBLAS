#include <cassert>
#include <iostream>
#include "../include/csr_matrix.hpp"
#include "../include/operations.hpp"

int main() {
    /*
    /Construct matrix:
    [ 10  0   3 ]
    [ 0   5   0 ]
    [ 2   0   7 ]
    */
    std::vector<int> rowptr = {0, 2, 3, 5};
    std::vector<int> colidx = {0, 2, 1, 0, 2};
    std::vector<double>      values = {10, 3, 5, 2, 7};

    CSRMatrix<double> A(3, 3, rowptr, colidx, values);
    {
        int count = 0;
        for (auto [col, value] : A.row(0)) {
            if (count == 0) {
                assert(col == 0 && value == 10); // Correct value from your previous example
            } else if (count == 1) {
                assert(col == 2 && value == 3.0);
            }
            count++;
        }
        assert(count == 2); // Check that exactly two elements were found
    }

    std::vector<double> x = {1, 2, 3};
    std::vector<double> y(3);
    slinalg::spmv(A, x, y);

    assert(y[0] == 19);
    assert(y[1] == 10);
    assert(y[2] == 23);

    std::cout << "CSR tests passed.\n";
    return 0;
}