#pragma once
#include <math.h>
#include <mpi.h>

typedef struct { double x, y; } Point2D;
typedef struct { double x, y, z; } Point3D;

static inline double dist_2D(Point2D a, Point2D b) {
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
}
static inline double dist_3D(Point3D a, Point3D b) {
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
}

typedef struct { double x, y, z; } Vec3D;

static inline Vec3D cross(Vec3D a, Vec3D b) {
    Vec3D cross_product = {
        .x = a.y*b.z - a.z*b.y,
        .y = a.z*b.x - a.x*b.z,
        .z = a.x*b.y - a.y*b.x
    };
    return cross_product;
}

static inline double dot(Vec3D a, Vec3D b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

typedef struct {
    Point3D pmax;
    Point3D pmin;
} AABB;

typedef struct {
    int dim;
    int n_verts;
    int n_prims;
    union {
        Point2D *verts2d; // (x,y) * n_verts
        Point3D *verts3d; // (x,y,z) * n_verts
    };
    int *prims;           // 2D: segments (s0,s1) * n_prims, 3D: tris (v0,v1,v2) * n_prims
} Mesh;

Mesh load_mesh(const char *path);
void free_mesh(Mesh *m);
void bcast_mesh(Mesh *m, int leader_rank, MPI_Comm comm);
void mesh_bbox(const Mesh *m, double *xmin, double *xmax, double *ymin, double *ymax, double *zmin, double *zmax);

AABB prim_bbox(const Mesh *m, int p);
Point3D centroid(const Mesh *m, int p);
