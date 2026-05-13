#pragma once
#include <stdbool.h>
#include "mesh.h"

int winding_number(const Mesh2D *m, Point2D p);
bool inside_mesh_2D(const Mesh2D *m, Point2D p);
