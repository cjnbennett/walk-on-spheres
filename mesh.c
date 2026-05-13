#include <ctype.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "mesh.h"

#define MAX_LINE_LEN 1024           // size of line buffer when reading mesh obj's

// minimal .obj -> Mesh2D parser
Mesh2D load_mesh_2D(const char *path) {
    FILE *fp = fopen(path, "r");
    
    // exit on invalid file read
    if (fp == NULL) {
        fprintf(stderr, "Could not read file %s\n", path);
        exit(1);
    }

    char line[MAX_LINE_LEN];

    // first pass for memory allocation
    int vert_count = 0;
    int seg_count = 0;
    while (fgets(line, MAX_LINE_LEN, fp) != NULL) {
        if (line[0] == 'v' && isspace((unsigned char)line[1])) vert_count++;
        else if (line[0] == 'l' && isspace((unsigned char)line[1])) {
            char *curr = line + 2;
            int vert;
            int shift;
            while (*curr != '\0' && sscanf(curr, " %d%n", &vert, &shift) == 1) {     // consume whitespace and next integer
                curr += shift;
                seg_count++;
            }
            seg_count--;
            // TODO - validate whether polyline loop is closed or not, auto-close if not
        }
    }
    rewind(fp);

    Mesh2D mesh = {
        .n_verts = vert_count,
        .n_segs = seg_count,
        .verts = malloc(vert_count * sizeof *mesh.verts),       // x1,y1,x2,y2,...
        .segs = malloc(2 * seg_count * sizeof *mesh.segs)       // [s0_start, s0_end, s1_start, s1_end, ...]
    };
    if (!mesh.verts || !mesh.segs) {
        fprintf(stderr, "Could not allocate memory while building mesh\n");
        exit(1);
    }

    // second pass for filling struct
    int v_idx = 0;
    int s_idx = 0;
    while (fgets(line, MAX_LINE_LEN, fp) != NULL) {
        if (line[0] == 'v' && isspace((unsigned char)line[1])) {
            // parse vertex coords
            double x, y, z;
            sscanf(line+2, "%lf %lf %lf", &x, &y, &z);
            // for a 2D mesh, we discard z
            Point2D p = { .x = x, .y = y };
            mesh.verts[v_idx++] = p;
        } else if (line[0] == 'l' && isspace((unsigned char)line[1])) {
            // parse polyline vertex ordering, emit consecutive pairs as segments
            char *curr = line + 2;
            int prev_vert = -1;     // 0-indexed previous vertex; -1 sentinel = first index of this polyline
            int vert;
            int shift;
            while (*curr != '\0' && sscanf(curr, " %d%n", &vert, &shift) == 1) {     // consume whitespace and next integer
                curr += shift;
                vert--;         // .obj vertices are 1-indexed => convert to 0-indexed
                if (prev_vert >= 0) {
                    mesh.segs[2*s_idx + 0] = prev_vert;
                    mesh.segs[2*s_idx + 1] = vert;
                    s_idx++;
                }
                prev_vert = vert;
            }
        }
    }

    fclose(fp);
    return mesh;
}

void free_mesh_2D(Mesh2D *m) {
    free(m->verts);
    free(m->segs);
}

void bcast_mesh_2D(Mesh2D *m, int leader_rank, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    MPI_Bcast(&m->n_verts, 1, MPI_INT, leader_rank, comm);
    MPI_Bcast(&m->n_segs, 1, MPI_INT, leader_rank, comm);

    if (rank != leader_rank) {
        // allocate memory
        m->verts = malloc(m->n_verts * sizeof *m->verts);
        m->segs = malloc(2 * m->n_segs * sizeof *m->segs);
        if (!m->verts || !m->segs) MPI_Abort(comm, 1);
    }

    MPI_Bcast(m->verts, 2 * m->n_verts, MPI_DOUBLE, leader_rank, comm); // broadcasts double x, y directly rather than Point2D struct
    MPI_Bcast(m->segs, 2 * m->n_segs, MPI_INT, leader_rank, comm);
}
