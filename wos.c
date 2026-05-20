#include <float.h>
#include <stdio.h>
#include "bvh.h"
#include "fastmath.h"
#include "mesh.h"
#include "prng.h"
#include "wos.h"

// weight factors for the screened WoS
double screening_factor_2D(double alpha, double radius) {
    double prod = radius * alpha;
    if (prod < 1e-8) return 1.0;
    return 1.0 / bessel_I0(prod);
}
double screening_factor_3D(double alpha, double radius) {
    double prod = radius * alpha;
    if (prod < 1e-8) return 1.0;
    return prod / sinh(prod);
}

// perform a singular step of the walk exiting on a sphere
Point2D step_2D(Sphere2D sphere) {
    // for standard WoS, the exit distribution is uniform
    double theta = prng_unit() * 2*M_PI;
    Point2D exit = {
        .x = sphere.centre.x + sphere.radius * cos(theta),
        .y = sphere.centre.y + sphere.radius * sin(theta)
    };
    return exit;
}
Point3D step_3D(Sphere3D sphere) {
    // sample uniformly from surface
    double u = prng_unit();
    double v = prng_unit();
    double z = 2*u-1;
    double phi = 2*M_PI*v;
    double r = sqrt(1-z*z);

    Point3D exit = {
        .x = sphere.centre.x + sphere.radius*r*cos(phi),
        .y = sphere.centre.y + sphere.radius*r*sin(phi),
        .z = sphere.centre.z + sphere.radius*z
    };
    return exit;
}

// draw a sample uniformly from the volume of a sphere, e.g. for sampling the source contribution
Point2D sphere_sample_2D(Sphere2D sphere) {
    double theta = prng_unit() * 2*M_PI;
    double r = sphere.radius * sqrt(prng_unit());

    Point2D sp = {
        .x = sphere.centre.x + r * cos(theta),
        .y = sphere.centre.y + r * sin(theta)
    };
    return sp;
}
Point3D sphere_sample_3D(Sphere3D sphere) {
    double u = prng_unit();
    double v = prng_unit();
    double w = prng_unit();
    double z = 2*u-1;
    double phi = 2*M_PI*v;
    double surf_fac = sqrt(1 - z*z);
    double vol_fac = sphere.radius * cbrt(w);

    Point3D sp = {
        .x = sphere.centre.x + vol_fac * surf_fac * cos(phi),
        .y = sphere.centre.y + vol_fac * surf_fac * sin(phi),
        .z = sphere.centre.z + vol_fac * z,
    };
    return sp;
}

// perform Monte Carlo walk-on-spheres (2D) to estimate u(p0) at a point p0
// source : function pointer to an inhomogeneous source term f(x,y)
// boundary : function pointer to Dirichlet boundary condition g(x,y)
double wos_2D(const BVH *bvh,
              Point2D p0,
              double (*source)(Point2D),
              double alpha,
              double (*boundary)(Point2D),
              double (*green)(Sphere2D,Point2D,Point2D),
              int N_walks,
              double epsilon)
{
    double mean = 0.0;

    Point2D p = p0;
    Point2D np;
    double r0 = bvh_npq(bvh, p, &np);

    for (int i = 0; i < N_walks; i++) {
        p = p0;
        double r = r0;
        Sphere2D sphere = {
            .centre = p,
            .radius = r
        };

        double sample = 0.0;
        double weight = 1.0;

        while (sphere.radius > epsilon) {
            // step includes source accumulation and screening factor
            Point2D sp = sphere_sample(sphere);
            sample += weight * sphere_volume(sphere) * source(sp) * green(sphere, p, sp);
            weight *= screening_factor_2D(alpha, sphere.radius);
            
            // iterate
            p = step(sphere);
            r = bvh_npq(bvh, p, &np);
            sphere = (Sphere2D){
                .centre = p,
                .radius = r
            };
        }

        sample += weight * boundary(np);
        mean += (sample - mean) / (i+1);    // Welford mean
    }

    return mean;
}


double wos_3D(const BVH *bvh,
              Point3D p0,
              double (*source)(Point3D),
              double alpha,
              double (*boundary)(Point3D),
              double (*green)(Sphere3D,Point3D,Point3D),
              int N_walks,
              double epsilon)
{
    double mean = 0.0;

    Point3D p = p0;
    Point3D np;
    double r0 = bvh_npq(bvh, p, &np);

    for (int i = 0; i < N_walks; i++) {
        p = p0;
        double r = r0;
        Sphere3D sphere = {
            .centre = p,
            .radius = r
        };

        double sample = 0.0;
        double weight = 1.0;

        while (sphere.radius > epsilon) {
            // step includes source accumulation and screening factor
            // TODO: alternate (more computationally efficient) method for asborption: Russian roulette
            // i.e. Bernoulli trial to kill walks early. Screening accumulation generalises better to
            // non-absorbing cases.
            Point3D sp = sphere_sample(sphere);
            sample += weight * sphere_volume(sphere) * source(sp) * green(sphere, p, sp);
            weight *= screening_factor_3D(alpha, sphere.radius);
            
            // iterate
            p = step(sphere);
            r = bvh_npq(bvh, p, &np);
            sphere = (Sphere3D){
                .centre = p,
                .radius = r
            };
        }

        sample += weight * boundary(np);
        mean += (sample - mean) / (i+1);    // Welford mean
    }

    return mean;
}
