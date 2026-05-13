#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include "mesh.h"

// check which side of a line segment a point p is on using signed triangle area
// > 0 : left, < 0 : right, = 0 : collinear
static inline double signed_tri_area(Point2D s0, Point2D s1, Point2D p) {
    return (s1.x - s0.x)*(p.y - s0.y) - (p.x - s0.x)*(s1.y - s0.y);
}

// Sunday's winding number algorithm for point-in-polygon tests (2D)
int winding_number(const Mesh *m, Point2D p) {
    int wn = 0;

    for (int s = 0; s < m->n_prims; s++) {   // naive O(n_prims) - can be improved using BVH
        // endpoints of segment
        Point2D s0 = m->verts2d[m->prims[2*s + 0]];
        Point2D s1 = m->verts2d[m->prims[2*s + 1]];

        if (s0.y <= p.y) {
            if (s1.y > p.y && signed_tri_area(s0,s1,p) > 0) wn++;   // s0->s1 crosses up through p
        } else {
            if (s1.y <= p.y && signed_tri_area(s0,s1,p) < 0) wn--;  // s0->s1 crosses down through p
        }
    }

    return wn;
}

// Möller-Trumbore ray-triangle intersection algorithm
bool moller_trumbore(Point3D p, Vec3D ray, Point3D a, Point3D b, Point3D c) {
    // TODO - not very robust (boundary points occasionally excluded)
    const double EPSILON = 1e-9;
    Vec3D edge1 = { .x = b.x - a.x, .y = b.y - a.y, .z = b.z - a.z };
    Vec3D edge2 = { .x = c.x - a.x, .y = c.y - a.y, .z = c.z - a.z };

    Vec3D rxe2 = cross(ray, edge2);
    double det = dot(edge1, rxe2);
    if (fabs(det) <= EPSILON) return false;     // ray parallel to tri

    Vec3D s = { .x = p.x - a.x, .y = p.y - a.y, .z = p.z - a.z };
    double u = dot(s, rxe2) / det;
    if (u < -EPSILON || u-1 > EPSILON) return false;    // ray outside edge2 bounds

    Vec3D sxe1 = cross(s, edge1);
    double v = dot(ray, sxe1) / det;
    if (v < -EPSILON || u+v-1 > EPSILON) return false;  // ray outside edge1 bounds

    // ray intersects tri - find intersection
    double t = dot(edge2, sxe1) / det;  // t is the distance between p and the intersection
    if (t > EPSILON) {  // ray intersection
        return true;
    } else {    // line intersection, but not ray intersection
        return false;
    }
}

// aggregate # ray-triangle intersections over entire Mesh (3D)
int count_ray_tri_intersect(const Mesh *m, Point3D p) {
    int intersects = 0;

    double u = (double)rand() / RAND_MAX;
    double v = (double)rand() / RAND_MAX;
    double z = 2*u-1;
    double phi = 2*M_PI*v;
    double r = sqrt(1-z*z);

    Vec3D ray = { .x = r*cos(phi), .y = r*sin(phi), .z = z };
    // TODO - test for edge cases on intersection of multiple faces (potentially over/under-counted), re-randomise ray
    for (int s = 0; s < m->n_prims; s++) {
        // vertices of tri
        Point3D v0 = m->verts3d[m->prims[3*s + 0]];
        Point3D v1 = m->verts3d[m->prims[3*s + 1]];
        Point3D v2 = m->verts3d[m->prims[3*s + 2]];

        if (moller_trumbore(p, ray, v0, v1, v2)) intersects++;
    }
    return intersects;
}

bool inside_mesh_2D(const Mesh *m, Point2D p) {
    return winding_number(m, p) != 0;
}

bool inside_mesh_3D(const Mesh *m, Point3D p) {
    const int N_RAYS = 3;

    int total = 0;
    for (int i = 0; i < N_RAYS; i++) {
        int hits = count_ray_tri_intersect(m, p);
        total += (hits & 1);
    }
    return total > N_RAYS/2;  // majority vote of N_RAYS for robustness
}
