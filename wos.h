#pragma once
#include <math.h>
#include "bvh.h"
#include "mesh.h"

// spheres for WoS
typedef struct {
    Point2D centre;
    double radius;
} Sphere2D;
typedef struct {
    Point3D centre;
    double radius;
} Sphere3D;

static inline double sphere_volume_2D(Sphere2D sphere) {
    return M_PI*sphere.radius*sphere.radius;
}
static inline double sphere_volume_3D(Sphere3D sphere) {
    double rcubed = sphere.radius*sphere.radius*sphere.radius;
    return 4.0*M_PI*rcubed/3.0;
}
// static dispatch for sphere volume
#define sphere_volume(sphere) _Generic((sphere), Sphere2D: sphere_volume_2D, Sphere3D: sphere_volume_3D)((sphere))

double screening_factor_2D(double alpha, double radius);
double screening_factor_3D(double alpha, double radius);

Point2D step_2D(Sphere2D sphere);
Point3D step_3D(Sphere3D sphere);
// static dispatch for step
#define step(sphere) _Generic((sphere), Sphere2D: step_2D, Sphere3D: step_3D)((sphere))

Point2D sphere_sample_2D(Sphere2D sphere);
Point3D sphere_sample_3D(Sphere3D sphere);
// static dispatch for sphere sample
#define sphere_sample(sphere) _Generic((sphere), Sphere2D: sphere_sample_2D, Sphere3D: sphere_sample_3D)((sphere))

double wos_2D(const BVH *bvh, Point2D p0, double (*source)(Point2D), double alpha, double (*boundary)(Point2D), double (*green)(Sphere2D,Point2D,Point2D), int N_walks, double epsilon);
double wos_3D(const BVH *bvh, Point3D p0, double (*source)(Point3D), double alpha, double (*boundary)(Point3D), double (*green)(Sphere3D,Point3D,Point3D), int N_walks, double epsilon);
// static dispatch for WoS
#define wos(bvh, p0, source, alpha, boundary, green, N_walks, epsilon) _Generic((p0), Point2D: wos_2D, Point3D: wos_3D)((bvh), (p0), (source), (alpha), (boundary), (green), (N_walks), (epsilon))

