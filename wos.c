#include <stdlib.h>
#include "mesh.h"
#include "npq.h"
#include "wos.h"

// perform a singular step of the walk exiting on a sphere
Point2D step_2D(Sphere2D sphere) {
    // for standard WoS, the exit distribution is uniform
    double theta = (double)rand() / (double)RAND_MAX * 2*M_PI;
    Point2D exit = {
        .x = sphere.centre.x + sphere.radius * cos(theta),
        .y = sphere.centre.y + sphere.radius * sin(theta)
    };
    return exit;
}
Point3D step_3D(Sphere3D sphere) {
    // sample uniformly from surface
    double u = (double)rand() / RAND_MAX;
    double v = (double)rand() / RAND_MAX;
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
    double theta = (double)rand() / (double)RAND_MAX * 2*M_PI;
    double r = sphere.radius * sqrt((double)rand() / (double)RAND_MAX);

    Point2D sp = {
        .x = sphere.centre.x + r * cos(theta),
        .y = sphere.centre.y + r * sin(theta)
    };
    return sp;
}
Point3D sphere_sample_3D(Sphere3D sphere) {
    double u = (double)rand() / RAND_MAX;
    double v = (double)rand() / RAND_MAX;
    double w = (double)rand() / RAND_MAX;
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
double wos_2D(const Mesh *Ω,
              Point2D p0,
              double (*source)(Point2D),
              double (*boundary)(Point2D),
              double (*green)(Sphere2D,Point2D,Point2D),
              int N_walks,
              double epsilon)
{
    double mean = 0.0;

    for (int i = 0; i < N_walks; i++) {
        Point2D p = p0;
        Point2D np;
        double r = npq_mesh(Ω, p, &np);
        Sphere2D sphere = {
            .centre = p,
            .radius = r
        };

        double sample = 0.0;

        while (sphere.radius > epsilon) {
            // for a PDE containing a source term (e.g. Poisson equation), we must accumulate a source
            // contribution for non-exiting walks
            Point2D sp = sphere_sample(sphere);
            sample += sphere_volume(sphere) * source(sp) * green(sphere, p, sp);
            
            // iterate
            p = step(sphere);
            r = npq_mesh(Ω, p, &np);
            sphere = (Sphere2D){
                .centre = p,
                .radius = r
            };
        }

        sample += boundary(np);
        mean += (sample - mean) / (i+1);    // Welford mean
    }

    return mean;
}


double wos_3D(const Mesh *Ω,
              Point3D p0,
              double (*source)(Point3D),
              double (*boundary)(Point3D),
              double (*green)(Sphere3D,Point3D,Point3D),
              int N_walks,
              double epsilon)
{
    double mean = 0.0;

    for (int i = 0; i < N_walks; i++) {
        Point3D p = p0;
        Point3D np;
        double r = npq_mesh(Ω, p, &np);
        Sphere3D sphere = {
            .centre = p,
            .radius = r
        };

        double sample = 0.0;

        while (sphere.radius > epsilon) {
            // for a PDE containing a source term (e.g. Poisson equation), we must accumulate a source
            // contribution for non-exiting walks
            Point3D sp = sphere_sample(sphere);
            sample += sphere_volume(sphere) * source(sp) * green(sphere, p, sp);
            
            // iterate
            p = step(sphere);
            r = npq_mesh(Ω, p, &np);
            sphere = (Sphere3D){
                .centre = p,
                .radius = r
            };
        }

        sample += boundary(np);
        mean += (sample - mean) / (i+1);    // Welford mean
    }

    return mean;
}
