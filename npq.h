#pragma once
#include "mesh.h"

double npq_seg(Point2D s0, Point2D s1, Point2D p, Point2D *nearest);
double npq_mesh_2D(const Mesh *m, Point2D p, Point2D *nearest);
double eberly(Point3D p, Point3D v0, Point3D v1, Point3D v2, Point3D *nearest);
double npq_mesh_3D(const Mesh *m, Point3D p, Point3D *nearest);

// static dispatch for mesh nearest point queries based on dimension
#define npq_mesh(m, p, nearest) _Generic((p), Point2D: npq_mesh_2D, Point3D: npq_mesh_3D)((m), (p), (nearest))
