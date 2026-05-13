#include <ctype.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "mesh.h"

#define MAX_LINE_LEN 1024           // size of line buffer when reading mesh .obj files

// minimal .obj -> Mesh parser, supports 2D and 3D meshes
Mesh load_mesh(const char *path) {
    FILE *fp = fopen(path, "r");
    
    // exit on invalid file read
    if (fp == NULL) {
        fprintf(stderr, "Could not read file %s\n", path);
        exit(1);
    }

    char line[MAX_LINE_LEN];

    // first pass for memory allocation
    int vert_count = 0;
    int prim_count = 0;
    bool has_l = false, has_f = false;
    while (fgets(line, MAX_LINE_LEN, fp) != NULL) {
        if (line[0] == 'v' && isspace((unsigned char)line[1])) vert_count++;
        else if (line[0] == 'l' && isspace((unsigned char)line[1])) {
            has_l = true;
            char *curr = line + 2;
            int vert;
            int shift;
            int first_vert = -1, last_vert = -1;
            int idx_count = 0;
            while (sscanf(curr, " %d%n", &vert, &shift) == 1) {
                curr += shift;
                if (first_vert < 0) first_vert = vert;
                last_vert = vert;
                idx_count++;
            }
            prim_count += idx_count - 1;
            if (idx_count >= 2 && first_vert != last_vert) {    // open polyline -> auto-close
                fprintf(stderr, "Warning: open polyline in %s auto-closed\n", path);
                prim_count++;
            }
        } else if (line[0] == 'f' && isspace((unsigned char)line[1])) {
            has_f = true;
            // only tris supported
            char *curr = line + 2;
            int fv_count = 0;
            while (*curr) {
                while (*curr && isspace((unsigned char)*curr)) curr++;
                if (!*curr) break;
                fv_count++;
                while (*curr && !isspace((unsigned char)*curr)) curr++;
            }
            if (fv_count != 3) {
                fprintf(stderr, "Mesh %s: face with %d vertices - only tris supported\n", path, fv_count);
                exit(1);
            }
            prim_count++;
        }
    }
    rewind(fp);

    if (has_l == has_f) {
        fprintf(stderr, "Could not parse mesh %s: conflicting dimension\n", path);
        exit(1);
    }
    int dim = has_l ? 2 : 3;

    Mesh mesh;
    if (dim == 2)
        mesh = (Mesh){
            .dim = dim,
            .n_verts = vert_count,
            .n_prims = prim_count,
            .verts2d = malloc(vert_count * sizeof(Point2D)),       // x1,y1,x2,y2,...
            .prims = malloc(dim * prim_count * sizeof *mesh.prims)     // [s0_start, s0_end, s1_start, s1_end, ...]
        };
    else
        mesh = (Mesh){
            .dim = dim,
            .n_verts = vert_count,
            .n_prims = prim_count,
            .verts3d = malloc(vert_count * sizeof(Point3D)),
            .prims = malloc(dim * prim_count * sizeof *mesh.prims)
        };
    if (!mesh.verts2d || !mesh.prims) {
        fprintf(stderr, "Could not allocate memory while building mesh\n");
        exit(1);
    }

    // second pass for filling struct
    int v_idx = 0;
    int p_idx = 0;
    while (fgets(line, MAX_LINE_LEN, fp) != NULL) {
        if (line[0] == 'v' && isspace((unsigned char)line[1])) {
            // parse vertex coords
            double x, y, z;
            sscanf(line+2, "%lf %lf %lf", &x, &y, &z);
            if (dim == 2) {
                // for a 2D mesh, we discard z
                Point2D p = { .x = x, .y = y };
                mesh.verts2d[v_idx++] = p;
            } else {
                Point3D p = { .x = x, .y = y, .z = z };
                mesh.verts3d[v_idx++] = p;
            }
        } else if (line[0] == 'l' && isspace((unsigned char)line[1])) {
            // parse polyline vertex ordering, emit consecutive pairs as segments
            char *curr = line + 2;
            int first_vert = -1, prev_vert = -1;
            int vert;
            int shift;
            while (sscanf(curr, " %d%n", &vert, &shift) == 1) {
                curr += shift;
                vert--;         // .obj vertices are 1-indexed => convert to 0-indexed
                if (first_vert < 0) first_vert = vert;
                if (prev_vert >= 0) {
                    mesh.prims[2*p_idx + 0] = prev_vert;
                    mesh.prims[2*p_idx + 1] = vert;
                    p_idx++;
                }
                prev_vert = vert;
            }
            // auto-close: if last vertex differs from first, emit closing segment
            if (prev_vert >= 0 && prev_vert != first_vert) {
                mesh.prims[2*p_idx + 0] = prev_vert;
                mesh.prims[2*p_idx + 1] = first_vert;
                p_idx++;
            }
        } else if (line[0] == 'f' && isspace((unsigned char)line[1])) {
            // parse face vertices
            int a, b, c;
            sscanf(line+2, "%d %d %d", &a, &b, &c);
            a--; b--; c--;      // .obj vertices are 1-indexed => convert to 0-index
            mesh.prims[3*p_idx + 0] = a;
            mesh.prims[3*p_idx + 1] = b;
            mesh.prims[3*p_idx + 2] = c;
            p_idx++;
        }
    }

    fclose(fp);
    return mesh;
}

// AABB for entire mesh
void mesh_bbox(const Mesh *m, double *xmin, double *xmax, double *ymin, double *ymax, double *zmin, double *zmax)
{
    double xl, xu, yl, yu, zl, zu;
    if (m->dim == 2) {
        xl = xu = m->verts2d[0].x;
        yl = yu = m->verts2d[0].y;
        zl = zu = 0.0;
        for (int i = 1; i < m->n_verts; i++) {
            Point2D v = m->verts2d[i];
            if (v.x < xl) xl = v.x;
            else if (v.x > xu) xu = v.x;
            if (v.y < yl) yl = v.y;
            else if (v.y > yu) yu = v.y;
        }
    } else {
        xl = xu = m->verts3d[0].x;
        yl = yu = m->verts3d[0].y;
        zl = zu = m->verts3d[0].z;
        for (int i = 1; i < m->n_verts; i++) {
            Point3D v = m->verts3d[i];
            if (v.x < xl) xl = v.x; else if (v.x > xu) xu = v.x;
            if (v.y < yl) yl = v.y; else if (v.y > yu) yu = v.y;
            if (v.z < zl) zl = v.z; else if (v.z > zu) zu = v.z;
        }
    }
    if (xmin) *xmin = xl;
    if (xmax) *xmax = xu;
    if (ymin) *ymin = yl;
    if (ymax) *ymax = yu;
    if (zmin) *zmin = zl;
    if (zmax) *zmax = zu;
}

void free_mesh(Mesh *m) {
    if (m->dim == 2) free(m->verts2d);
    else free(m->verts3d);
    free(m->prims);
}

void bcast_mesh(Mesh *m, int leader_rank, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    MPI_Bcast(&m->dim, 1, MPI_INT, leader_rank, comm);
    MPI_Bcast(&m->n_verts, 1, MPI_INT, leader_rank, comm);
    MPI_Bcast(&m->n_prims, 1, MPI_INT, leader_rank, comm);

    if (rank != leader_rank) {
        // allocate memory
        if (m->dim == 2) m->verts2d = malloc(m->n_verts * sizeof(Point2D));
        else m->verts3d = malloc(m->n_verts * sizeof(Point3D));
        m->prims = malloc(m->dim * m->n_prims * sizeof *m->prims);
        if (!(m->verts2d || m->verts3d) || !m->prims) MPI_Abort(comm, 1);
    }

    if (m->dim == 2) MPI_Bcast(m->verts2d, m->dim * m->n_verts, MPI_DOUBLE, leader_rank, comm); // broadcasts double x, y directly rather than Point2D/3D
    else MPI_Bcast(m->verts3d, m->dim * m->n_verts, MPI_DOUBLE, leader_rank, comm);
    MPI_Bcast(m->prims, m->dim * m->n_prims, MPI_INT, leader_rank, comm);
}
