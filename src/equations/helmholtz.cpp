// Helmholtz equation: ∆u + k²u = -f on a domain Ω, with boundary u = g on ∂Ω
// source f is optional
#include <cmath>
#include "equations.hpp"
#include "wos/fastmath.hpp"
#include "wos/mesh.hpp"
#include "wos/runner.hpp"
#include "wos/wos.hpp"

using namespace wos;

namespace {

struct Helmholtz2D {
    [[maybe_unused]] static constexpr bool has_source = false;  // switch to true for inhomogeneous Helmholtz
    [[maybe_unused]] static constexpr bool has_screening = true;
    static constexpr double k = 1.0;    // first resonance ~ 2.405 on unit circle

    double source(Point2D p) const {
        (void)p;
        return 0.0;
    }

    double boundary(Point2D p) const {
        (void)p;
        return 1.0; // uniform forcing
    }

    // Green's function for Helmholtz operator on 2D spherical domain
    double green(Sphere2D sphere, Point2D x, Point2D y) const {
        double r = dist(x, y);
        double prod = r * k;
        double prod_R = sphere.radius * k;
        return (bessel_J0(prod) * bessel_Y0(prod_R) / bessel_J0(prod_R) - bessel_Y0(prod)) / 4.0;
    }

    // weight factor: 1 / J_0(k*r) for the Helmholtz eq in 2D
    double screening_factor(double radius) const {
        double prod = radius * k;
        if (prod < 1e-8) return 1.0;
        return 1.0 / bessel_J0(prod);
    }
};

struct Helmholtz3D {
    [[maybe_unused]] static constexpr bool has_source = false;  // switch to true for inhomogeneous Helmholtz
    [[maybe_unused]] static constexpr bool has_screening = true;
    static constexpr double k = 1.0;

    double source(Point3D p) const {
        (void)p;
        return 0.0;
    }

    double boundary(Point3D p) const {
        (void)p;
        return 1.0;
    }

    // Green's function for Helmholtz operator on 3D spherical domain
    double green(Sphere3D sphere, Point3D x, Point3D y) const {
        double r = dist(x, y);
        double prod = r * k;
        double prod_R = sphere.radius * k;
        return (std::sin(prod_R - prod)) / (4*M_PI * r * std::sin(prod_R));
    }

    // weight factor
    double screening_factor(double radius) const {
        double prod = radius * k;
        if (prod < 1e-8) return 1.0;
        return prod / std::sin(prod);
    }
};

int run_2D(int rank, int size, const char *mesh, int Nx, int Ny, int Nz, int N_walks, double eps) {
    return run<2>(rank, size, mesh, Nx, Ny, Nz, N_walks, eps, Helmholtz2D{});
}
int run_3D(int rank, int size, const char *mesh, int Nx, int Ny, int Nz, int N_walks, double eps) {
    return run<3>(rank, size, mesh, Nx, Ny, Nz, N_walks, eps, Helmholtz3D{});
}

}   // anonymous namespace

// external linkage
const Equation helmholtz = {
    "helmholtz",
    run_2D,
    run_3D,
};
