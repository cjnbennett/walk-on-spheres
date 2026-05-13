#pragma once
#include "mesh.h"

double npq_seg(Point2D s0, Point2D s1, Point2D p, Point2D *nearest);
double npq_mesh_2D(const Mesh2D *m, Point2D p, Point2D *nearest);
