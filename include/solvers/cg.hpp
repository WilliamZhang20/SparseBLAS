#pragma once
#include <vector>
#include <cmath>
#include <optional>
#include <span>
#include "../operations.hpp"
#include "../csr_matrix.hpp"

// More info at https://stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf

namespace slinalg {

struct CGResult {
    int iters;
    double final_residual;
    bool converged;
};

struct CGParams {
    int max_iters = 1000;
    double eps = 1e-8;
};

template <class T>
inline CGResult conjugate_gradient (
    const CSRMatrix<T>& A,
    std::span<const double> b,
    stD::span<double> x,
    CGParams params = {})
{
    std::size_t n = A.nrows();
    std::vector<double> r(n), p(n), Ap(n);

    // r = b - A x
    spmv(A, x, r);
    for (std::size_t i = 0; i < n; ++i)
        r[i] = b[i] - r[i];

    p = r;
    double rsold = dot(r, r);

    CGResult result{0, std::sqrt(rsold), false};

    for (int k = 0; k < params.max_iters; ++k) {
        spmv(A, p, Ap);
        double alpha = rsold / dot(p, Ap);

        // x = x + alpha p
        axpy(alpha, p, x);

        // r = r - alpha Ap
        axpy(-alpha, Ap, r);

        double rsnew = dot(r, r);
        result.final_residual = std::sqrt(rsnew);
        result.iters = k + 1;

        if (result.final_residual < params.tol) {
            result.converged = true;
            break;
        }

        double beta = rsnew / rsold;
        rsold = rsnew;

        // p = r + beta p
        for (std::size_t i = 0; i < n; ++i)
            p[i] = r[i] + beta * p[i];
    }

    return result;
}

};