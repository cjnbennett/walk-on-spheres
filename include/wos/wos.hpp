#pragma once
#include "wos/bvh.hpp"
#include "wos/mesh.hpp"

namespace wos {

struct Sphere2D { Point2D centre; double radius; };
struct Sphere3D { Point3D centre; double radius; };

// helper to select appropriately dimensioned sphere for Sphere
template<int N> struct SphereSelector;
template<> struct SphereSelector<2> { using type = Sphere2D; };
template<> struct SphereSelector<3> { using type = Sphere3D; };
template<int N> using Sphere = typename SphereSelector<N>::type;

inline double sphere_volume(Sphere2D s) {
    return M_PI * s.radius * s.radius;
}
inline double sphere_volume(Sphere3D s) {
    double rcubed = s.radius * s.radius * s.radius;
    return 4.0 * M_PI * rcubed / 3.0;
}

Point2D step(Sphere2D sphere);
Point3D step(Sphere3D sphere);

Point2D sphere_sample(Sphere2D sphere);
Point3D sphere_sample(Sphere3D sphere);

// Monte Carlo walk-on-spheres estimator for u(p0)
// Eq should specify source, screening, boundary, etc. (as appropriate)
template<int N, typename Eq>
double wos_run(const BVH<N> &bvh, Point<N> p0, const Eq &eq, int N_walks, double epsilon) {
    Point<N> p = p0;
    Point<N> np;
    double r0 = bvh_npq(bvh, p, &np);

    double mean = 0.0;

    for (int i = 0; i < N_walks; i++) {
        p = p0;
        double r = r0;
        Sphere<N> sphere{p, r};

        double sample = 0.0;
        double weight = 1.0;

        while (sphere.radius > epsilon) {
            if constexpr (Eq::has_source) {
                Point<N> sp = sphere_sample(sphere);
                sample += weight * sphere_volume(sphere) * eq.source(sp) * eq.green(sphere, p, sp);
            }
            if constexpr (Eq::has_screening) {
                weight *= eq.screening_factor(sphere.radius);
            }

            p = step(sphere);
            r = bvh_npq(bvh, p, &np);
            sphere = Sphere<N>{p, r};
        }

        sample += weight * eq.boundary(np);
        mean += (sample - mean) / (i+1);    // Welford mean
    }

    return mean;
}

}
