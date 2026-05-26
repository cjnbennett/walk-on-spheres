#include <cmath>
#include "wos/mesh.hpp"
#include "wos/prng.hpp"
#include "wos/wos.hpp"

namespace wos {

// perform a single step of the walk exiting on a sphere
Point2D step(Sphere2D sphere) {
    // for standard WoS, the exit distribution is uniform
    double theta = prng_unit() * 2*M_PI;
    return Point2D{
        sphere.centre.x + sphere.radius * std::cos(theta),
        sphere.centre.y + sphere.radius * std::sin(theta)
    };
}
Point3D step(Sphere3D sphere) {
    // sample uniformly from surface
    double u = prng_unit();
    double v = prng_unit();
    double z = 2*u-1;
    double phi = 2*M_PI*v;
    double r = std::sqrt(1-z*z);

    return Point3D{
        sphere.centre.x + sphere.radius*r*std::cos(phi),
        sphere.centre.y + sphere.radius*r*std::sin(phi),
        sphere.centre.z + sphere.radius*z
    };
}

// draw a sample uniformly from the volume of a sphere, e.g. for sampling the source contribution
Point2D sphere_sample(Sphere2D sphere) {
    double theta = prng_unit() * 2*M_PI;
    double r = sphere.radius * std::sqrt(prng_unit());

    return Point2D{
        sphere.centre.x + r * std::cos(theta),
        sphere.centre.y + r * std::sin(theta)
    };
}
Point3D sphere_sample(Sphere3D sphere) {
    double u = prng_unit();
    double v = prng_unit();
    double w = prng_unit();
    double z = 2*u-1;
    double phi = 2*M_PI*v;
    double surf_fac = std::sqrt(1 - z*z);
    double vol_fac = sphere.radius * std::cbrt(w);

    return Point3D{
        sphere.centre.x + vol_fac * surf_fac * std::cos(phi),
        sphere.centre.y + vol_fac * surf_fac * std::sin(phi),
        sphere.centre.z + vol_fac * z,
    };
}

}
