#pragma once
#include "wos/mesh.hpp"

namespace wos {

int winding_number(const Mesh<2> &m, Point2D p);
bool moller_trumbore(Point3D p, Vec3D ray, Point3D a, Point3D b, Point3D c);
int count_ray_tri_intersect(const Mesh<3> &m, Point3D p);

bool inside_mesh(const Mesh<2> &m, Point2D p);
bool inside_mesh(const Mesh<3> &m, Point3D p);

}
