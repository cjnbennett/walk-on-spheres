#include <math.h>
#include "mesh.h"

// nearest point query for a line segment. Returns the nearest point distance and writes the nearest point to *nearest
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

// nearest point query on 2D Mesh. Returns the nearest point distance and writes the nearest point to *nearest
double npq_mesh_2D(const Mesh *m, Point2D p, Point2D *nearest) {
    double closest_d_sq = INFINITY;         // closest squared distance so far

    for (int s = 0; s < m->n_prims; s++) {   // naive O(n_prims) - can be improved using BVH
        // endpoints of segment
        Point2D s0 = m->verts2d[m->prims[2*s + 0]];
        Point2D s1 = m->verts2d[m->prims[2*s + 1]];

        Point2D seg_nearest;  // store closest point on current seg
        double d_sq = npq_seg(s0, s1, p, &seg_nearest);

        if (d_sq < closest_d_sq) {
            closest_d_sq = d_sq;
            *nearest = seg_nearest; 
        }
    }

    return sqrt(closest_d_sq);
}

// nearest point query for a tri via Eberley's algorithm. Returns the nearest point squared distance and writes the nearest point to *nearest
double eberly(Point3D p, Point3D v0, Point3D v1, Point3D v2, Point3D *nearest) {
    Vec3D edge1 = { .x = v1.x - v0.x, .y = v1.y - v0.y, .z = v1.z - v0.z };
    Vec3D edge2 = { .x = v2.x - v0.x, .y = v2.y - v0.y, .z = v2.z - v0.z };
    Vec3D BP = { .x = v0.x - p.x, .y = v0.y - p.y, .z = v0.z - p.z };
    double a = dot(edge1, edge1);
    double b = dot(edge1, edge2);
    double c = dot(edge2, edge2);
    double d = dot(edge1, BP);
    double e = dot(edge2, BP);

    double s = b*e - c*d;
    double t = b*d - a*e;
    double det = a*c - b*b;

    if (s + t <= det) {
        if (s < 0) {
            if (t < 0) {
                // region 4
                if (d < 0) {
                    t = 0;
                    if (-d >= a) {
                        s = 1;
                    } else {
                        s = -d/a;
                    }
                } else {
                    s = 0;
                    if (e >= 0) {
                        t = 0;
                    } else if (-e >= c) {
                        t = 1;
                    } else {
                        t = -e/c;
                    }
                }
            } else {
                // region 3
                s = 0;
                if (e >= 0) {
                    t = 0;
                } else if (-e >= c) {
                    t = 1;
                } else {
                    t = -e/c;
                }
            }
        } else if (t < 0) {
            // region 5
            t = 0;
            if (d >= 0) {
                s = 0;
            } else if (-d >= a) {
                s = 1;
            } else {
                s = -d/a;
            }
        } else {
            // region 0
            s /= det;
            t /= det;
        }
    } else {
        if (s < 0) {
            // region 2
            double bd = b+d;
            double ce = c+e;
            if (ce > bd) {
                double numer = ce - bd;
                double denom = a - 2*b + c;
                if (numer >= denom) {
                    s = 1;
                } else {
                    s = numer / denom;
                }
                t = 1-s;
            } else {
                s = 0;
                if (ce <= 0) {
                    t = 1;
                } else if (e >= 0) {
                    t = 0;
                } else {
                    t = -e/c;
                }
            }
        } else if (t < 0) {
            // region 6
            double be = b+e;
            double ad = a+d;
            if (ad > be) {
                double numer = ad - be;
                double denom = a - 2*b + c;
                if (numer >= denom) {
                    t = 1;
                } else {
                    t = numer / denom;
                }
                s = 1-t;
            } else {
                t = 0;
                if (ad <= 0) {
                    s = 1;
                } else if (d >= 0) {
                    s = 0;
                } else {
                    s = -d/a;
                }
            }
        } else {
            // region 1
            double numer = (c+e) - (b+d);
            if (numer <= 0) {
                s = 0;
            } else {
                double denom = a - 2*b + c;
                if (numer >= denom) {
                    s = 1;
                } else {
                    s = numer / denom;
                }
            }
            t = 1-s;
        }
    }

    *nearest = (Point3D) {
        .x = v0.x + s*edge1.x + t*edge2.x,
        .y = v0.y + s*edge1.y + t*edge2.y,
        .z = v0.z + s*edge1.z + t*edge2.z
    };

    Vec3D displacement = {
        .x = nearest->x - p.x,
        .y = nearest->y - p.y,
        .z = nearest->z - p.z
    };
    return dot(displacement, displacement);
}

// nearest point query on 3D Mesh. Returns the nearest point distance and writes the nearest point to *nearest
double npq_mesh_3D(const Mesh *m, Point3D p, Point3D *nearest) {
    double closest_d_sq = INFINITY;         // closest squared distance so far

    for (int s = 0; s < m->n_prims; s++) {   // naive O(n_prims) - can be improved using BVH
        // endpoints of segment
        Point3D v0 = m->verts3d[m->prims[3*s + 0]];
        Point3D v1 = m->verts3d[m->prims[3*s + 1]];
        Point3D v2 = m->verts3d[m->prims[3*s + 2]];

        Point3D tri_nearest;  // store closest point on current tri
        double d_sq = eberly(p, v0, v1, v2, &tri_nearest);

        if (d_sq < closest_d_sq) {
            closest_d_sq = d_sq;
            *nearest = tri_nearest; 
        }
    }

    return sqrt(closest_d_sq);
}
