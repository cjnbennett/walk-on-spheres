// This file implements a simple walk-on-spheres (WoS) to solve the Poisson equation ∆u = -f on a
// domain Ω, with boundary condition u = g on ∂Ω. Here Ω is an arbitary 2D/3D mesh. To recover
// the Laplace equation, set f = 0.
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "bvh.h"
#include "grid.h"
#include "hash.h"
#include "hdf5_io.h"
#include "mesh.h"
#include "inside.h"
#include "wos.h"

// f: source function, defined on Ω
double f_2D(Point2D p) {
    // a bump centred at (0.65, 0) with radius 0.15
    return sqrt((p.x-0.65)*(p.x-0.65) + p.y*p.y) <= 0.15 ? 10.0 : 0.0;
}
double f_3D(Point3D p) {
    return 0.0;
}

// g: boundary value function for u(x,y), defined for (x,y) ∈ ∂Ω
double g_2D(Point2D p) {
    double r = sqrt(p.x*p.x + p.y*p.y);
    return r > 0.7 ? p.x*p.x - p.y*p.y      // outer: saddle
                   : 0.0;                   // inner: zero
}
double g_3D(Point3D p) {
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z) * signbit(p.x);
}

// Green's function for PDE operator on spherical domain ("harmonic")
double G_2D(Sphere2D sphere, Point2D x, Point2D y) {
    // Green's function for Laplace operator on 2D spherical domain
    return log(sphere.radius / dist_2D(x,y)) / (2*M_PI);
}
double G_3D(Sphere3D sphere, Point3D x, Point3D y) {
    // Green's function for Laplace operator on 3D spherical domain
    double r = dist_3D(x,y);
    return (sphere.radius - r) / (4*M_PI * r * sphere.radius);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // default params
    char *mesh_filename = "meshes/annulus.obj";   // path to domain mesh .obj on disk
    int Nx = 32, Ny = 32, Nz = 32;                // grid resolution per axis
    int N_walks = 1e4;                            // number of walks simulated per point
    double epsilon = 1e-6;                        // tolerance for WoS to boundary

    // positional args: poisson [Nx Ny] [Nz] [mesh.obj]
    if (rank == 0) {
        if (argc >= 3) {
            Nx = atoi(argv[1]);
            Ny = atoi(argv[2]);
        }
        if (argc >= 4) {
            // try parse argv[3] as Nz, otherwise mesh.obj
            char *end;
            long val = strtol(argv[3], &end, 10);
            if (end != argv[3] && *end == '\0' && val > 0) {
                Nz = (int)val;
                if (argc >= 5) mesh_filename = argv[4];
            } else {
                mesh_filename = argv[3];
            }
        }
    }

    MPI_Bcast(&Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_walks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    uint64_t seed;
    if (rank == 0) seed = (uint64_t)time(NULL);
    MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    srand((unsigned int)(seed ^ splitmix64((uint64_t)rank)));   // note: this is probably a really bad way to set the seed

    // load mesh as domain
    Mesh Ω = {0};
    if (rank == 0) {
        printf("Reading mesh from disk (%s)...\n", mesh_filename);
        Ω = load_mesh(mesh_filename);
        printf("Finished reading. Broadcasting mesh...\n");
    }
    bcast_mesh(&Ω, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Finished broadcasting.\n");

    if (rank == 0) printf("Building BVH...\n");
    BVH *bvh = build_bvh(&Ω);
    if (rank == 0) printf("Finished BVH build.\n");

    // setup domain discretisation
    if (Ω.dim == 2) Nz = 1;
    Grid grid = { .Nx = Nx, .Ny = Ny, .Nz = Nz };
    mesh_bbox(&Ω, &grid.xmin, &grid.xmax, &grid.ymin, &grid.ymax, &grid.zmin, &grid.zmax);

    // 3D Cartesian decomposition: split ranks across x, y, z axes
    int dims[3] = { 0, 0, 0 };
    if (Ω.dim == 2) dims[2] = 1;
    MPI_Dims_create(size, 3, dims);

    // map flat rank -> (cx, cy, cz) coords in the rank grid
    int cx = rank / (dims[1] * dims[2]);
    int cy = (rank / dims[2]) % dims[1];
    int cz = rank % dims[2];

    int x_start = (grid.Nx * cx) / dims[0];
    int x_end = (grid.Nx * (cx + 1)) / dims[0];
    int y_start = (grid.Ny * cy) / dims[1];
    int y_end = (grid.Ny * (cy + 1)) / dims[1];
    int z_start = (grid.Nz * cz) / dims[2];
    int z_end = (grid.Nz * (cz + 1)) / dims[2];
    int block_Nx = x_end - x_start;
    int block_Ny = y_end - y_start;
    int block_Nz = z_end - z_start;

    if (rank == 0) {
        printf("Parallelisation: %d x %d x %d ranks across (Nx=%d, Ny=%d, Nz=%d)\n", dims[0], dims[1], dims[2], grid.Nx, grid.Ny, grid.Nz);
    }

    // u[x][y][z]
    int block_doubles = block_Nx * block_Ny * block_Nz;
    double (*u)[block_Ny][block_Nz] = block_doubles > 0 ? malloc(block_doubles * sizeof(double)) : NULL;
    if (block_doubles > 0 && !u) {
        fprintf(stderr, "malloc failed for rank %d, aborting\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) printf("Beginning walk-on-spheres...\n");

    // run WoS on points in block
    for (int i = 0; i < block_Nx; i++) {
        double x = grid.Nx > 1 ? grid.xmin + (grid.xmax - grid.xmin) * (x_start + i) / (grid.Nx - 1) : 0.0;
        for (int j = 0; j < block_Ny; j++) {
            double y = grid.Ny > 1 ? grid.ymin + (grid.ymax - grid.ymin) * (y_start + j) / (grid.Ny - 1) : 0.0;
            for (int k = 0; k < block_Nz; k++) {
                double z = grid.Nz > 1 ? grid.zmin + (grid.zmax - grid.zmin) * (z_start + k) / (grid.Nz - 1) : 0.0;
                // sample only points in the domain
                if (Ω.dim == 2) {
                    Point2D p0 = { x, y };
                    u[i][j][k] = inside_mesh(&Ω, p0) ? wos(bvh, p0, &f_2D, &g_2D, &G_2D, N_walks, epsilon) : NAN;
                } else {
                    Point3D p0 = { x, y, z };
                    u[i][j][k] = inside_mesh(&Ω, p0) ? wos(bvh, p0, &f_3D, &g_3D, &G_3D, N_walks, epsilon) : NAN;
                }
            }
        }
    }

    if (rank == 0) printf("Finished walk-on-spheres. Writing results...\n");

    wos_write_hdf5("poisson_wos.h5", grid, x_start, y_start, z_start, block_Nx, block_Ny, block_Nz, (const double *)u);
    if (rank == 0) printf("Finished writing.\n");

    free_bvh(bvh);
    free_mesh(&Ω);
    free(u);
    MPI_Finalize();
    return 0;
}
