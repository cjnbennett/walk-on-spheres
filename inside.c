#include <stdbool.h>
#include "mesh.h"

// check which side of a line segment a point p is on using signed triangle area
// > 0 : left, < 0 : right, = 0 : collinear
static inline double signed_tri_area(Point2D s0, Point2D s1, Point2D p) {
    return (s1.x - s0.x)*(p.y - s0.y) - (p.x - s0.x)*(s1.y - s0.y);
}

// Sunday's winding number algorithm for point-in-polygon tests
int winding_number(const Mesh2D *m, Point2D p) {
    int wn = 0;

    for (int s = 0; s < m->n_segs; s++) {   // naive O(n_segs) - can be improved using BVH
        // endpoints of segment
        Point2D s0 = m->verts[m->segs[s]];
        Point2D s1 = m->verts[m->segs[s+1]];

        if (s0.y <= p.y) {
            if (s1.y > p.y && signed_tri_area(s0,s1,p) > 0) wn++;   // s0->s1 crosses up through p
        } else {
            if (s1.y <= p.y && signed_tri_area(s0,s1,p) < 0) wn--;  // s0->s1 crosses down through p
        }
    }

    return wn;
}

// wrapper function
bool inside_mesh_2D(const Mesh2D *m, Point2D p) {
    return winding_number(m, p) != 0;
}
