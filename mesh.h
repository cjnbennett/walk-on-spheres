#pragma once
#include <math.h>
#include <mpi.h>

typedef struct {
    double x;
    double y;
} Point2D;

static inline double dist_2D(Point2D a, Point2D b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

typedef struct {
    int n_verts;
    int n_segs;
    Point2D *verts; // memory equivalent to double *verts: x1,y1,x2,y2,...,xn,yn
    int *segs;      // v0,v1,...,vn,v0 - TODO: doesn't generalise to multiple polylines
} Mesh2D;

Mesh2D load_mesh_2D(const char *path);
void free_mesh_2D(Mesh2D *m);
void bcast_mesh_2D(Mesh2D *m, int leader_rank, MPI_Comm comm);
