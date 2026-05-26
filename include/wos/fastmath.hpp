#pragma once

namespace wos {

// inlined min, max (no NaN support)
static inline double dmin(double a, double b) { return a < b ? a : b; }
static inline double dmax(double a, double b) { return a > b ? a : b; }

// cylindrical Bessel function
// TODO: not very fast. Polynomial approximation better
static inline double bessel_I0(double x) {
    double half_x_sq = 0.25 * x * x;

    double term = 1.0;      // 0th order initialisation
    double sum = 1.0;

    for (int k = 1; k < 200; k++) {
        term *= half_x_sq / (double)(k * k);
        if (term < 1e-8) return sum;    // convergence - exit
        sum += term;
    }

    return sum;
}

}
