#include <math.h>
#include "mesh.h"

// nearest point query for a 2D line segment. Returns the nearest point distance and writes the nearest point to *nearest
double npq_seg(Point2D s0, Point2D s1, Point2D p, Point2D *nearest) {
    double vx = s1.x - s0.x;
    double vy = s1.y - s0.y;
    double wx = p.x - s0.x;
    double wy = p.y - s0.y;

    double vv = vx*vx + vy*vy;  // |v|^2
    double wv = wx*vx + wy*vy;  // w·v

    double par = (vv > 0.0) ? wv / vv : 0.0;    // parallel projection of w onto v
    // clamp projection to endpoints of line seg
    if (par < 0.0) par = 0.0;
    if (par > 1.0) par = 1.0;
    
    nearest->x = s0.x + par*vx;
    nearest->y = s0.y + par*vy;

    // distance p->nearest
    double dx = p.x - nearest->x;
    double dy = p.y - nearest->y;
    return dx*dx + dy*dy;   // d^2
}

// nearest point query on Mesh2D. Returns the nearest point distance and writes the nearest point to *nearest
double npq_mesh_2D(const Mesh2D *m, Point2D p, Point2D *nearest) {
    double closest_d_sq = INFINITY;         // closest squared distance so far

    for (int s = 0; s < m->n_segs; s++) {   // naive O(n_segs) - can be improved using BVH
        // endpoints of segment
        Point2D s0 = m->verts[m->segs[2*s + 0]];
        Point2D s1 = m->verts[m->segs[2*s + 1]];

        Point2D seg_nearest;  // store closest point on current seg
        double d_sq = npq_seg(s0, s1, p, &seg_nearest);

        if (d_sq < closest_d_sq) {
            closest_d_sq = d_sq;
            *nearest = seg_nearest; 
        }
    }

    return sqrt(closest_d_sq);
}
