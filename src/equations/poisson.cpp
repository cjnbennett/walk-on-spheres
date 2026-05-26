// Poisson equation: ∆u = -f on a domain Ω, with boundary u = g on ∂Ω
#include <cmath>
#include "equations.hpp"
#include "wos/mesh.hpp"
#include "wos/runner.hpp"
#include "wos/wos.hpp"

using namespace wos;

namespace {

struct Poisson2D {
    [[maybe_unused]] static constexpr bool has_source = true;
    [[maybe_unused]] static constexpr bool has_screening = false;

    double source(Point2D p) const {
        // a bump centred at (0.45, 0) with radius 0.5
        return std::sqrt((p.x-0.45)*(p.x-0.45) + p.y*p.y) <= 0.5 ? 5.0 : 0.0;
    }

    double boundary(Point2D p) const {
        double r = std::sqrt(p.x*p.x + p.y*p.y);
        return r > 0.7 ? p.x*p.x - p.y*p.y      // outer: saddle
                       : 0.0;                   // inner: zero
    }

    // Green's function for Laplace operator on 2D spherical domain
    double green(Sphere2D sphere, Point2D x, Point2D y) const {
        return std::log(sphere.radius / dist(x, y)) / (2*M_PI);
    }
};

struct Poisson3D {
    [[maybe_unused]] static constexpr bool has_source = true;
    [[maybe_unused]] static constexpr bool has_screening = false;

    double source(Point3D p) const {
        (void)p;
        return 0.0;
    }

    double boundary(Point3D p) const {
        // combination of spherical harmonics
        double r_sq = p.x*p.x + p.y*p.y;
        double y50  = p.z * (8*p.z*p.z*p.z*p.z - 40*p.z*p.z*r_sq + 15*r_sq*r_sq);
        double y5m3 = (8*p.z*p.z - r_sq) * p.y * (3*p.x*p.x - p.y*p.y);
        return y50 + 4.0 * y5m3;
    }

    // Green's function for Laplace operator on 3D spherical domain
    double green(Sphere3D sphere, Point3D x, Point3D y) const {
        double r = dist(x, y);
        return (sphere.radius - r) / (4*M_PI * r * sphere.radius);
    }
};

int run_2D(int rank, int size, const char *mesh, int Nx, int Ny, int Nz, int N_walks, double eps) {
    return run<2>(rank, size, mesh, Nx, Ny, Nz, N_walks, eps, Poisson2D{});
}
int run_3D(int rank, int size, const char *mesh, int Nx, int Ny, int Nz, int N_walks, double eps) {
    return run<3>(rank, size, mesh, Nx, Ny, Nz, N_walks, eps, Poisson3D{});
}

}   // anonymous namespace

// external linkage
const Equation poisson = {
    "poisson",
    run_2D,
    run_3D,
};
