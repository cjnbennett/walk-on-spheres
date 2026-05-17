#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include "bvh.h"
#include "npq.h"
#include "mesh.h"

#define STACK_MAX 64

typedef struct {
    AABB bbox;
    bool leaf;
    int idx;    // if (leaf) => points to mesh->prims[idx], if (!leaf) => points to right child BVHNode at nodes[idx]
    EberlyCache eberly_cache;   // only needs to be populated if (leaf) and if 3D BVH
} BVHNode;

typedef struct {   // temporary data structure used to construct an efficient BVH
    AABB bbox;
    Point3D centroid;     // for 2D meshes, z is 0
    int prim_idx;
} BVHBuildNode;

struct BVH {
    const Mesh *mesh;
    BVHNode *nodes;
    int *prims;
};

// returns lower bound for squared distance to nearest point in aabb from p
static inline double aabb_d_sq_2D(const AABB *aabb, Point2D p) {
    double dx = fmax(0.0, fmax(aabb->pmin.x - p.x, p.x - aabb->pmax.x));
    double dy = fmax(0.0, fmax(aabb->pmin.y - p.y, p.y - aabb->pmax.y));
    return dx*dx + dy*dy;
}
static inline double aabb_d_sq_3D(const AABB *aabb, Point3D p) {
    double dx = fmax(0.0, fmax(aabb->pmin.x - p.x, p.x - aabb->pmax.x));
    double dy = fmax(0.0, fmax(aabb->pmin.y - p.y, p.y - aabb->pmax.y));
    double dz = fmax(0.0, fmax(aabb->pmin.z - p.z, p.z - aabb->pmax.z));
    return dx*dx + dy*dy + dz*dz;
}

static AABB union_bbox(const BVHBuildNode *centroids, int start, int end) {
    AABB u_bbox = centroids[start].bbox;
    for (int i = start + 1; i < end; i++) {
        AABB c_bbox = centroids[i].bbox;
        u_bbox.pmin.x = fmin(u_bbox.pmin.x, c_bbox.pmin.x);
        u_bbox.pmin.y = fmin(u_bbox.pmin.y, c_bbox.pmin.y);
        u_bbox.pmin.z = fmin(u_bbox.pmin.z, c_bbox.pmin.z);
        u_bbox.pmax.x = fmax(u_bbox.pmax.x, c_bbox.pmax.x);
        u_bbox.pmax.y = fmax(u_bbox.pmax.y, c_bbox.pmax.y);
        u_bbox.pmax.z = fmax(u_bbox.pmax.z, c_bbox.pmax.z);
    }
    return u_bbox;
}

static int qsort_axis;   // module-scope; build is single-threaded per rank

static int cmp_centroid(const void *a, const void *b) {
    const BVHBuildNode *ea = a, *eb = b;
    double ca, cb;
    switch (qsort_axis) {
        case 0:
            ca = ea->centroid.x;
            cb = eb->centroid.x;
            break;
        case 1:
            ca = ea->centroid.y;
            cb = eb->centroid.y;
            break;
        case 2:
        default:
            ca = ea->centroid.z;
            cb = eb->centroid.z;
            break;
    }
    return (ca > cb) - (ca < cb);
}

static void nth_element(BVHBuildNode *centroids, int start, int end, int mid, int axis) {
    (void)mid;
    qsort_axis = axis;
    // TODO - for now use qsort, but in future replace with quickselect / introselect
    qsort(&centroids[start], end - start, sizeof *centroids, cmp_centroid);
}

// recursively build bvh subtree for nodes [start,end)
static int build_subtree(BVH *bvh, BVHBuildNode *centroids, int start, int end, int *node_ptr) {
    int span = end - start;
    AABB node_bbox = union_bbox(centroids, start, end);
    int node = (*node_ptr)++;  // reserve a node in the BVH
    bvh->nodes[node].bbox = node_bbox;
    bvh->nodes[node].leaf = false;

    if (span == 1) {    // leaf node
        // node bbox already set above
        bvh->nodes[node].leaf = true;
        bvh->nodes[node].idx = start;
        bvh->prims[start] = centroids[start].prim_idx;

        const Mesh *m = bvh->mesh;
        if (m->dim == 3) {
            int prim_idx = bvh->prims[start];

            Point3D v0 = m->verts3d[m->prims[3*prim_idx + 0]];
            Point3D v1 = m->verts3d[m->prims[3*prim_idx + 1]];
            Point3D v2 = m->verts3d[m->prims[3*prim_idx + 2]];

            Vec3D edge1 = { .x = v1.x - v0.x, .y = v1.y - v0.y, .z = v1.z - v0.z };
            Vec3D edge2 = { .x = v2.x - v0.x, .y = v2.y - v0.y, .z = v2.z - v0.z };
            double a = dot(edge1, edge1);
            double b = dot(edge1, edge2);
            double c = dot(edge2, edge2);
            double det = a*c - b*b;
            double denom = a - 2*b + c;

            bvh->nodes[node].eberly_cache = (EberlyCache){
                .v0=v0,
                .edge1=edge1,
                .edge2=edge2,
                .a=a,
                .b=b,
                .c=c,
                .det=det,
                .denom=denom
            };
        }

        return node;
    }

    double ex = node_bbox.pmax.x - node_bbox.pmin.x;
    double ey = node_bbox.pmax.y - node_bbox.pmin.y;
    double ez = node_bbox.pmax.z - node_bbox.pmin.z;
    int axis = (ex >= ey) ? (ex >= ez ? 0 : 2) : (ey >= ez ? 1 : 2);
    int mid = (start+end)/2;
    nth_element(centroids, start, end, mid, axis);

    int left = build_subtree(bvh, centroids, start, mid, node_ptr);
    assert(left == node + 1);
    int right = build_subtree(bvh, centroids, mid, end, node_ptr);
    bvh->nodes[node].idx = right;
    return node;
}

BVH *build_bvh(const Mesh *m) {
    BVH *bvh = malloc(sizeof *bvh);
    bvh->mesh = m;
    bvh->nodes = malloc((2*m->n_prims-1) * sizeof *bvh->nodes); // 2*n_prims-1 nodes
    bvh->prims = malloc(m->n_prims * sizeof *bvh->prims);
    
    // setup scratch array
    BVHBuildNode *centroids = malloc(m->n_prims * sizeof(BVHBuildNode));
    for (int p = 0; p < m->n_prims; p++) {
        centroids[p] = (BVHBuildNode) {
            .bbox=prim_bbox(m, p),
            .centroid=centroid(m, p),
            .prim_idx=p
        };
    }

    // recursively build the bvh tree
    int node_ptr = 0;
    build_subtree(bvh, centroids, 0, m->n_prims, &node_ptr);

    free(centroids);    // free scratch array
    return bvh;
}

// query nearest point to p on bvh->mesh boundary, return distance, write nearest point to *nearest
double bvh_npq_2D(const BVH *bvh, Point2D p, Point2D *nearest) {
    const Mesh *m = bvh->mesh;
    double closest_d_sq = INFINITY;         // closest squared distance so far

    int stack[STACK_MAX];
    int sp = 0;

    stack[sp++] = 0;    // push root node to stack

    while (sp > 0) {
        int curr = stack[--sp];

        const BVHNode *curr_node = &bvh->nodes[curr];

        if (aabb_d_sq_2D(&curr_node->bbox, p) >= closest_d_sq) continue;   // prune more distant nodes

        if (curr_node->leaf) {      // accumulate nearest point query on primitive
            int s = bvh->prims[curr_node->idx];
            // endpoints of segment
            Point2D s0 = m->verts2d[m->prims[2*s + 0]];
            Point2D s1 = m->verts2d[m->prims[2*s + 1]];

            Point2D seg_nearest;  // store closest point on current seg
            double d_sq = npq_seg(s0, s1, p, &seg_nearest);

            if (d_sq < closest_d_sq) {
                closest_d_sq = d_sq;
                *nearest = seg_nearest; 
            }
        } else {
            // push nearer child first
            int l = curr+1;
            int r = curr_node->idx;
            double l_d_sq = aabb_d_sq_2D(&bvh->nodes[l].bbox, p);
            double r_d_sq = aabb_d_sq_2D(&bvh->nodes[r].bbox, p);

            if (l_d_sq <= r_d_sq) {
                stack[sp++] = l;
                stack[sp++] = r;
            } else {
                stack[sp++] = r;
                stack[sp++] = l;
            }
        }
    }

    return sqrt(closest_d_sq);
}

// query nearest point to p on bvh->mesh boundary, return distance, write nearest point to *nearest
double bvh_npq_3D(const BVH *bvh, Point3D p, Point3D *nearest) {
    double closest_d_sq = INFINITY;         // closest squared distance so far

    int stack[STACK_MAX];
    int sp = 0;

    stack[sp++] = 0;    // push root node to stack

    while (sp > 0) {
        int curr = stack[--sp];
        const BVHNode *curr_node = &bvh->nodes[curr];

        if (aabb_d_sq_3D(&curr_node->bbox, p) >= closest_d_sq) continue;   // prune more distant nodes

        if (curr_node->leaf) {      // accumulate nearest point query on primitive
            Point3D tri_nearest;
            double d_sq = eberly(p, &tri_nearest, &curr_node->eberly_cache);

            if (d_sq < closest_d_sq) {
                closest_d_sq = d_sq;
                *nearest = tri_nearest; 
            }
        } else {
            // push nearer child first
            int l = curr+1;
            int r = curr_node->idx;
            double l_d_sq = aabb_d_sq_3D(&bvh->nodes[l].bbox, p);
            double r_d_sq = aabb_d_sq_3D(&bvh->nodes[r].bbox, p);

            if (l_d_sq <= r_d_sq) {
                stack[sp++] = l;
                stack[sp++] = r;
            } else {
                stack[sp++] = r;
                stack[sp++] = l;
            }
        }
    }

    return sqrt(closest_d_sq);
}

void free_bvh(BVH *bvh) {
    free(bvh->nodes);
    free(bvh->prims);
    free(bvh);
}
