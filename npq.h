#pragma once
#include "mesh.h"

double npq_seg(Point2D s0, Point2D s1, Point2D p, Point2D *nearest);
double npq_naive_2D(const Mesh *m, Point2D p, Point2D *nearest);
double eberly(Point3D p, Point3D v0, Point3D v1, Point3D v2, Point3D *nearest);
double npq_naive_3D(const Mesh *m, Point3D p, Point3D *nearest);

// static dispatch
#define npq_naive(m, p, nearest) _Generic((p), Point2D: npq_naive_2D, Point3D: npq_naive_3D)((m), (p), (nearest))
