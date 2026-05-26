#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include "wos/bvh.hpp"
#include "wos/fastmath.hpp"
#include "wos/mesh.hpp"
#include "wos/npq.hpp"

namespace wos {

namespace {

constexpr int STACK_MAX = 64;
constexpr int LEAF_SIZE = 4;

template<int N>
struct BVHBuildNode {   // temporary data used during construction
    AABB<N> bbox;
    Point<N> centroid;
    int prim_idx;
};

inline double aabb_d_sq(const AABB<2> &aabb, Point2D p) {
    double dx = dmax(0.0, dmax(aabb.pmin.x - p.x, p.x - aabb.pmax.x));
    double dy = dmax(0.0, dmax(aabb.pmin.y - p.y, p.y - aabb.pmax.y));
    return dx*dx + dy*dy;
}
inline double aabb_d_sq(const AABB<3> &aabb, Point3D p) {
    double dx = dmax(0.0, dmax(aabb.pmin.x - p.x, p.x - aabb.pmax.x));
    double dy = dmax(0.0, dmax(aabb.pmin.y - p.y, p.y - aabb.pmax.y));
    double dz = dmax(0.0, dmax(aabb.pmin.z - p.z, p.z - aabb.pmax.z));
    return dx*dx + dy*dy + dz*dz;
}

template<int N>
AABB<N> union_bbox(const std::vector<BVHBuildNode<N>> &centroids, int start, int end) {
    AABB<N> u = centroids[start].bbox;
    for (int i = start + 1; i < end; i++) {
        const AABB<N> &c = centroids[i].bbox;
        u.pmin.x = dmin(u.pmin.x, c.pmin.x);
        u.pmin.y = dmin(u.pmin.y, c.pmin.y);
        u.pmax.x = dmax(u.pmax.x, c.pmax.x);
        u.pmax.y = dmax(u.pmax.y, c.pmax.y);
        if constexpr (N == 3) {
            u.pmin.z = dmin(u.pmin.z, c.pmin.z);
            u.pmax.z = dmax(u.pmax.z, c.pmax.z);
        }
    }
    return u;
}

template<int N>
void sort_by_axis(std::vector<BVHBuildNode<N>> &centroids, int start, int end, int axis) {
    // TODO: replace with std::nth_element
    std::sort(centroids.begin() + start, centroids.begin() + end,
              [axis](const BVHBuildNode<N> &a, const BVHBuildNode<N> &b) {
                  if constexpr (N == 2) {
                      return (axis == 0) ? (a.centroid.x < b.centroid.x) : (a.centroid.y < b.centroid.y);
                  } else {
                      switch (axis) {
                          case 0:
                            return a.centroid.x < b.centroid.x;
                          case 1:
                            return a.centroid.y < b.centroid.y;
                          default:
                            return a.centroid.z < b.centroid.z;
                      }
                  }
              });
}

template<int N>
int build_subtree(BVH<N> &bvh, std::vector<BVHBuildNode<N>> &centroids, int start, int end, int &node_ptr) {
    int span = end - start;
    AABB<N> node_bbox = union_bbox(centroids, start, end);
    int node = node_ptr++;
    bvh.nodes[node].bbox = node_bbox;

    if (span <= LEAF_SIZE) {
        bvh.nodes[node].prim_count = span;
        bvh.nodes[node].idx = start;

        const Mesh<N> *m = bvh.mesh;
        for (int i = 0; i < span; i++) {
            int prim_idx = centroids[start + i].prim_idx;
            bvh.prims[start + i] = prim_idx;

            if constexpr (N == 3) {
                // pre-compute values for Eberly to save work in the hot-loop
                Point3D v0 = m->verts[m->prims[3*prim_idx + 0]];
                Point3D v1 = m->verts[m->prims[3*prim_idx + 1]];
                Point3D v2 = m->verts[m->prims[3*prim_idx + 2]];

                Vec3D edge1 = Vec3D{v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
                Vec3D edge2 = Vec3D{v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
                double a = dot(edge1, edge1);
                double b = dot(edge1, edge2);
                double c = dot(edge2, edge2);
                double det = a*c - b*b;
                double denom = a - 2*b + c;

                bvh.eberly_caches[start + i] = EberlyCache{v0, edge1, edge2, a, b, c, det, denom};
            }
        }

        return node;
    }

    bvh.nodes[node].prim_count = 0;    // internal node

    double ex = node_bbox.pmax.x - node_bbox.pmin.x;
    double ey = node_bbox.pmax.y - node_bbox.pmin.y;
    int axis;
    if constexpr (N == 2) {
        axis = (ex >= ey) ? 0 : 1;
    } else {
        double ez = node_bbox.pmax.z - node_bbox.pmin.z;
        axis = (ex >= ey) ? (ex >= ez ? 0 : 2) : (ey >= ez ? 1 : 2);
    }
    int mid = (start + end) / 2;
    sort_by_axis(centroids, start, end, axis);

    [[maybe_unused]] int left = build_subtree(bvh, centroids, start, mid, node_ptr);
    assert(left == node + 1);
    int right = build_subtree(bvh, centroids, mid, end, node_ptr);
    bvh.nodes[node].idx = right;
    return node;
}

}  // anonymous namespace

template<int N>
std::unique_ptr<BVH<N>> build_bvh(const Mesh<N> &m) {
    auto bvh = std::make_unique<BVH<N>>();
    bvh->mesh = &m;
    bvh->nodes.resize(2*m.n_prims() - 1);   // worst-case (LEAF_SIZE=1) node count
    bvh->prims.resize(m.n_prims());
    if constexpr (N == 3) bvh->eberly_caches.resize(m.n_prims());

    std::vector<BVHBuildNode<N>> centroids(m.n_prims());
    for (int p = 0; p < m.n_prims(); p++) {
        centroids[p] = BVHBuildNode<N>{prim_bbox(m, p), centroid(m, p), p};
    }

    int node_ptr = 0;
    build_subtree(*bvh, centroids, 0, m.n_prims(), node_ptr);

    return bvh;
}

double bvh_npq(const BVH<2> &bvh, Point2D p, Point2D *nearest) {
    const Mesh<2> *m = bvh.mesh;
    double closest_d_sq = INFINITY;

    int stack[STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int curr = stack[--sp];
        const BVHNode<2> &curr_node = bvh.nodes[curr];

        if (aabb_d_sq(curr_node.bbox, p) >= closest_d_sq) continue;

        if (curr_node.prim_count > 0) {
            for (int i = 0; i < curr_node.prim_count; i++) {
                int s = bvh.prims[curr_node.idx + i];
                Point2D s0 = m->verts[m->prims[2*s + 0]];
                Point2D s1 = m->verts[m->prims[2*s + 1]];

                Point2D seg_nearest;
                double d_sq = npq_seg(s0, s1, p, &seg_nearest);

                if (d_sq < closest_d_sq) {
                    closest_d_sq = d_sq;
                    *nearest = seg_nearest;
                }
            }
        } else {
            // push nearer child last so it's popped first (LIFO)
            int l = curr+1;
            int r = curr_node.idx;
            double l_d_sq = aabb_d_sq(bvh.nodes[l].bbox, p);
            double r_d_sq = aabb_d_sq(bvh.nodes[r].bbox, p);

            if (l_d_sq <= r_d_sq) {
                stack[sp++] = r;
                stack[sp++] = l;
            } else {
                stack[sp++] = l;
                stack[sp++] = r;
            }
        }
    }

    return std::sqrt(closest_d_sq);
}

double bvh_npq(const BVH<3> &bvh, Point3D p, Point3D *nearest) {
    double closest_d_sq = INFINITY;

    int stack[STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int curr = stack[--sp];
        const BVHNode<3> &curr_node = bvh.nodes[curr];

        if (aabb_d_sq(curr_node.bbox, p) >= closest_d_sq) continue;

        if (curr_node.prim_count > 0) {
            for (int i = 0; i < curr_node.prim_count; i++) {
                Point3D tri_nearest;
                double d_sq = eberly(p, &tri_nearest, &bvh.eberly_caches[curr_node.idx + i]);

                if (d_sq < closest_d_sq) {
                    closest_d_sq = d_sq;
                    *nearest = tri_nearest;
                }
            }
        } else {
            int l = curr+1;
            int r = curr_node.idx;
            double l_d_sq = aabb_d_sq(bvh.nodes[l].bbox, p);
            double r_d_sq = aabb_d_sq(bvh.nodes[r].bbox, p);

            if (l_d_sq <= r_d_sq) {
                stack[sp++] = r;
                stack[sp++] = l;
            } else {
                stack[sp++] = l;
                stack[sp++] = r;
            }
        }
    }

    return std::sqrt(closest_d_sq);
}

// explicit instantiations for 2D and 3D
template std::unique_ptr<BVH<2>> build_bvh<2>(const Mesh<2>&);
template std::unique_ptr<BVH<3>> build_bvh<3>(const Mesh<3>&);

}
