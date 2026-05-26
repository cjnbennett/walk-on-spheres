#pragma once
#include <cmath>
#include <cstdio>
#include <mpi.h>
#include "wos/bvh.hpp"
#include "wos/field_3D.hpp"
#include "wos/grid.hpp"
#include "wos/hdf5_io.hpp"
#include "wos/inside.hpp"
#include "wos/mesh.hpp"
#include "wos/wos.hpp"

namespace wos {

// MPI-parallel sweep of WoS over a grid covering the mesh's bounding box
template<int N, typename Eq>
int run(int rank, int size, const char *mesh_filename, int Nx, int Ny, int Nz,
        int N_walks, double epsilon, const Eq &eq) {
    Mesh<N> domain;
    if (rank == 0) {
        std::printf("Reading mesh from disk (%s)...\n", mesh_filename);
        domain = load_mesh<N>(mesh_filename);
        std::printf("Finished reading. Broadcasting mesh...\n");
    }
    bcast_mesh(domain, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Finished broadcasting.\n");

    if (rank == 0) std::printf("Building BVH...\n");
    auto bvh = build_bvh(domain);
    if (rank == 0) std::printf("Finished BVH build.\n");

    // setup domain discretisation
    if constexpr (N == 2) Nz = 1;
    Grid grid{Nx, Ny, Nz, 0,0, 0,0, 0,0};
    mesh_bbox(domain, &grid.xmin, &grid.xmax, &grid.ymin, &grid.ymax, &grid.zmin, &grid.zmax);

    // 3D Cartesian decomposition: split ranks across x, y, z axes
    int dims[3] = { 0, 0, 0 };
    if constexpr (N == 2) dims[2] = 1;
    MPI_Dims_create(size, 3, dims);

    int cx = rank / (dims[1] * dims[2]);
    int cy = (rank / dims[2]) % dims[1];
    int cz = rank % dims[2];

    int x_start = (grid.Nx * cx) / dims[0];
    int x_end   = (grid.Nx * (cx + 1)) / dims[0];
    int y_start = (grid.Ny * cy) / dims[1];
    int y_end   = (grid.Ny * (cy + 1)) / dims[1];
    int z_start = (grid.Nz * cz) / dims[2];
    int z_end   = (grid.Nz * (cz + 1)) / dims[2];
    int block_Nx = x_end - x_start;
    int block_Ny = y_end - y_start;
    int block_Nz = z_end - z_start;

    if (rank == 0) {
        std::printf("Parallelisation: %d x %d x %d ranks across (Nx=%d, Ny=%d, Nz=%d)\n",
                    dims[0], dims[1], dims[2], grid.Nx, grid.Ny, grid.Nz);
    }

    Field3D u(block_Nx, block_Ny, block_Nz);

    if (rank == 0) std::printf("Beginning walk-on-spheres...\n");

    for (int i = 0; i < block_Nx; i++) {
        double x = grid.Nx > 1 ? grid.xmin + (grid.xmax - grid.xmin) * (x_start + i) / (grid.Nx - 1) : 0.0;
        for (int j = 0; j < block_Ny; j++) {
            double y = grid.Ny > 1 ? grid.ymin + (grid.ymax - grid.ymin) * (y_start + j) / (grid.Ny - 1) : 0.0;
            for (int k = 0; k < block_Nz; k++) {
                double z = grid.Nz > 1 ? grid.zmin + (grid.zmax - grid.zmin) * (z_start + k) / (grid.Nz - 1) : 0.0;
                Point<N> p0;
                if constexpr (N == 2) {
                    p0 = Point2D{x, y};
                } else {
                    p0 = Point3D{x, y, z};
                }
                u(i, j, k) = inside_mesh(domain, p0) ? wos_run(*bvh, p0, eq, N_walks, epsilon) : NAN;
            }
        }
    }

    if (rank == 0) std::printf("Finished walk-on-spheres. Writing results...\n");

    wos_write_hdf5("wos.h5", grid, x_start, y_start, z_start, u);
    if (rank == 0) std::printf("Finished writing.\n");

    return 0;
}

}
