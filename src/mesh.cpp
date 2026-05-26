#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include "wos/mesh.hpp"

namespace wos {

constexpr int MAX_LINE_LEN = 1024;

int peek_mesh_dim(const char *path) {
    FILE *fp = std::fopen(path, "r");
    if (fp == nullptr) {
        std::fprintf(stderr, "Could not read file %s\n", path);
        std::exit(1);
    }

    char line[MAX_LINE_LEN];
    int dim = 0;
    while (std::fgets(line, MAX_LINE_LEN, fp) != nullptr) {
        if (line[0] == 'l' && std::isspace((unsigned char)line[1])) { dim = 2; break; }
        else if (line[0] == 'f' && std::isspace((unsigned char)line[1])) { dim = 3; break; }
    }

    std::fclose(fp);

    if (dim == 0) {
        std::fprintf(stderr, "Could not determine mesh dim in %s (no 'l' or 'f' entries)\n", path);
        std::exit(1);
    }

    return dim;
}

template<int N>
Mesh<N> load_mesh(const char *path) {
    FILE *fp = std::fopen(path, "r");
    if (fp == nullptr) {
        std::fprintf(stderr, "Could not read file %s\n", path);
        std::exit(1);
    }

    char line[MAX_LINE_LEN];

    // first pass for counts
    int vert_count = 0;
    int prim_count = 0;
    while (std::fgets(line, MAX_LINE_LEN, fp) != nullptr) {
        if (line[0] == 'v' && std::isspace((unsigned char)line[1])) {
            vert_count++;
        } else if constexpr (N == 2) {
            if (line[0] == 'l' && std::isspace((unsigned char)line[1])) {
                char *curr = line + 2;
                int vert, shift;
                int first_vert = -1, last_vert = -1;
                int idx_count = 0;
                while (std::sscanf(curr, " %d%n", &vert, &shift) == 1) {
                    curr += shift;
                    if (first_vert < 0) first_vert = vert;
                    last_vert = vert;
                    idx_count++;
                }
                prim_count += idx_count - 1;
                if (idx_count >= 2 && first_vert != last_vert) {
                    std::fprintf(stderr, "Warning: open polyline in %s auto-closed\n", path);
                    prim_count++;
                }
            }
        } else {
            if (line[0] == 'f' && std::isspace((unsigned char)line[1])) {
                // only tris supported
                char *curr = line + 2;
                int fv_count = 0;
                while (*curr) {
                    while (*curr && std::isspace((unsigned char)*curr)) curr++;
                    if (!*curr) break;
                    fv_count++;
                    while (*curr && !std::isspace((unsigned char)*curr)) curr++;
                }
                if (fv_count != 3) {
                    std::fprintf(stderr, "Mesh %s: face with %d vertices - only tris supported\n", path, fv_count);
                    std::exit(1);
                }
                prim_count++;
            }
        }
    }
    std::rewind(fp);

    Mesh<N> mesh;
    mesh.verts.resize(vert_count);
    mesh.prims.resize(N * prim_count);

    // second pass for filling
    int v_idx = 0;
    int p_idx = 0;
    while (std::fgets(line, MAX_LINE_LEN, fp) != nullptr) {
        if (line[0] == 'v' && std::isspace((unsigned char)line[1])) {
            double x, y, z;
            std::sscanf(line+2, "%lf %lf %lf", &x, &y, &z);
            if constexpr (N == 2) {
                mesh.verts[v_idx++] = Point2D{x, y};   // discard z
            } else {
                mesh.verts[v_idx++] = Point3D{x, y, z};
            }
        } else if constexpr (N == 2) {
            if (line[0] == 'l' && std::isspace((unsigned char)line[1])) {
                char *curr = line + 2;
                int first_vert = -1, prev_vert = -1;
                int vert, shift;
                while (std::sscanf(curr, " %d%n", &vert, &shift) == 1) {
                    curr += shift;
                    vert--;   // .obj is 1-indexed
                    if (first_vert < 0) first_vert = vert;
                    if (prev_vert >= 0) {
                        mesh.prims[2*p_idx + 0] = prev_vert;
                        mesh.prims[2*p_idx + 1] = vert;
                        p_idx++;
                    }
                    prev_vert = vert;
                }
                if (prev_vert >= 0 && prev_vert != first_vert) {
                    mesh.prims[2*p_idx + 0] = prev_vert;
                    mesh.prims[2*p_idx + 1] = first_vert;
                    p_idx++;
                }
            }
        } else {
            if (line[0] == 'f' && std::isspace((unsigned char)line[1])) {
                int a, b, c;
                std::sscanf(line+2, "%d %d %d", &a, &b, &c);
                a--; b--; c--;
                mesh.prims[3*p_idx + 0] = a;
                mesh.prims[3*p_idx + 1] = b;
                mesh.prims[3*p_idx + 2] = c;
                p_idx++;
            }
        }
    }

    std::fclose(fp);
    return mesh;
}

template<int N>
void bcast_mesh(Mesh<N> &m, int leader_rank, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int n_verts = m.n_verts();
    int n_prims = m.n_prims();
    MPI_Bcast(&n_verts, 1, MPI_INT, leader_rank, comm);
    MPI_Bcast(&n_prims, 1, MPI_INT, leader_rank, comm);

    if (rank != leader_rank) {
        m.verts.resize(n_verts);
        m.prims.resize(N * n_prims);
    }

    // verts: N doubles per vertex, broadcast contiguously
    MPI_Bcast(m.verts.data(), N * n_verts, MPI_DOUBLE, leader_rank, comm);
    MPI_Bcast(m.prims.data(), N * n_prims, MPI_INT,    leader_rank, comm);
}

template<int N>
void mesh_bbox(const Mesh<N> &m, double *xmin, double *xmax, double *ymin, double *ymax, double *zmin, double *zmax) {
    auto p0 = m.verts[0];
    double xl = p0.x, xu = p0.x;
    double yl = p0.y, yu = p0.y;
    double zl = 0.0, zu = 0.0;
    if constexpr (N == 3) {
        zl = p0.z;
        zu = p0.z;
    }

    for (int i = 1; i < m.n_verts(); i++) {
        auto v = m.verts[i];
        if (v.x < xl) xl = v.x; else if (v.x > xu) xu = v.x;
        if (v.y < yl) yl = v.y; else if (v.y > yu) yu = v.y;
        if constexpr (N == 3) {
            if (v.z < zl) zl = v.z;
            else if (v.z > zu) zu = v.z;
        }
    }
    if (xmin) *xmin = xl;
    if (xmax) *xmax = xu;
    if (ymin) *ymin = yl;
    if (ymax) *ymax = yu;
    if (zmin) *zmin = zl;
    if (zmax) *zmax = zu;
}

template<int N>
AABB<N> prim_bbox(const Mesh<N> &m, int p) {
    AABB<N> bbox{};
    if constexpr (N == 2) {
        Point2D s0 = m.verts[m.prims[2*p + 0]];
        Point2D s1 = m.verts[m.prims[2*p + 1]];
        bbox.pmin = Point2D{ std::fmin(s0.x, s1.x), std::fmin(s0.y, s1.y) };
        bbox.pmax = Point2D{ std::fmax(s0.x, s1.x), std::fmax(s0.y, s1.y) };
    } else {
        Point3D v0 = m.verts[m.prims[3*p + 0]];
        Point3D v1 = m.verts[m.prims[3*p + 1]];
        Point3D v2 = m.verts[m.prims[3*p + 2]];
        bbox.pmin = Point3D{
            std::fmin(std::fmin(v0.x, v1.x), v2.x),
            std::fmin(std::fmin(v0.y, v1.y), v2.y),
            std::fmin(std::fmin(v0.z, v1.z), v2.z),
        };
        bbox.pmax = Point3D{
            std::fmax(std::fmax(v0.x, v1.x), v2.x),
            std::fmax(std::fmax(v0.y, v1.y), v2.y),
            std::fmax(std::fmax(v0.z, v1.z), v2.z),
        };
    }
    return bbox;
}

template<int N>
Point<N> centroid(const Mesh<N> &m, int p) {
    if constexpr (N == 2) {
        Point2D s0 = m.verts[m.prims[2*p + 0]];
        Point2D s1 = m.verts[m.prims[2*p + 1]];
        return Point2D{ (s0.x+s1.x)/2.0, (s0.y+s1.y)/2.0 };
    } else {
        Point3D v0 = m.verts[m.prims[3*p + 0]];
        Point3D v1 = m.verts[m.prims[3*p + 1]];
        Point3D v2 = m.verts[m.prims[3*p + 2]];
        return Point3D{
            (v0.x+v1.x+v2.x)/3.0,
            (v0.y+v1.y+v2.y)/3.0,
            (v0.z+v1.z+v2.z)/3.0
        };
    }
}

// explicit instantiations for 2D and 3D
template Mesh<2> load_mesh<2>(const char *);
template Mesh<3> load_mesh<3>(const char *);
template void bcast_mesh<2>(Mesh<2>&, int, MPI_Comm);
template void bcast_mesh<3>(Mesh<3>&, int, MPI_Comm);
template void mesh_bbox<2>(const Mesh<2>&, double*, double*, double*, double*, double*, double*);
template void mesh_bbox<3>(const Mesh<3>&, double*, double*, double*, double*, double*, double*);
template AABB<2> prim_bbox<2>(const Mesh<2>&, int);
template AABB<3> prim_bbox<3>(const Mesh<3>&, int);
template Point2D centroid<2>(const Mesh<2>&, int);
template Point3D centroid<3>(const Mesh<3>&, int);

}
