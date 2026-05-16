#pragma once
#include "mesh.h"

typedef struct BVH BVH;

BVH *build_bvh(const Mesh *m);
void free_bvh(BVH *bvh);

double bvh_npq_2D(const BVH *bvh, Point2D p, Point2D *nearest);
double bvh_npq_3D(const BVH *bvh, Point3D p, Point3D *nearest);
#define bvh_npq(bvh, p, nearest) _Generic((p), Point2D: bvh_npq_2D, Point3D: bvh_npq_3D)((bvh), (p), (nearest))
