#pragma once
#include <stdbool.h>
#include "mesh.h"

int winding_number(const Mesh *m, Point2D p);
bool moller_trumbore(Point3D p, Vec3D ray, Point3D a, Point3D b, Point3D c);
int count_ray_tri_intersect(const Mesh *m, Point3D p);
bool inside_mesh_2D(const Mesh *m, Point2D p);
bool inside_mesh_3D(const Mesh *m, Point3D p);

// static dispatch for inside_mesh based on dimension
#define inside_mesh(m, p) _Generic((p), Point2D: inside_mesh_2D, Point3D: inside_mesh_3D)((m), (p))
