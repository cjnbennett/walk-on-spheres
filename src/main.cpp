// Walk-on-spheres solver launcher
// CLI: wos [equation] [Nx Ny] [Nz] [mesh.obj]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <mpi.h>
#include "equations/equations.hpp"
#include "wos/hash.hpp"
#include "wos/mesh.hpp"
#include "wos/prng.hpp"

using namespace wos;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // defaults
    const char *eq_name = "screened_poisson";
    const char *mesh_filename = "meshes/annulus.obj";
    int Nx = 32, Ny = 32, Nz = 32;
    int N_walks = 1'000;
    double epsilon = 1e-2;

    // positional args: wos [equation] [Nx Ny] [Nz] [mesh.obj]
    if (rank == 0) {
        if (argc >= 2) eq_name = argv[1];
        if (argc >= 4) {
            Nx = std::atoi(argv[2]);
            Ny = std::atoi(argv[3]);
        }
        if (argc >= 5) {
            // try parse argv[4] as Nz; if not numeric, treat as mesh filename
            char *end;
            long val = std::strtol(argv[4], &end, 10);
            if (end != argv[4] && *end == '\0' && val > 0) {
                Nz = (int)val;
                if (argc >= 6) mesh_filename = argv[5];
            } else {
                mesh_filename = argv[4];
            }
        }
    }

    MPI_Bcast(&Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_walks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int eq_idx = -1;
    if (rank == 0) {
        for (int i = 0; i < (int)(sizeof(equation_registry) / sizeof(*equation_registry)); i++) {
            if (std::strcmp(equation_registry[i]->name, eq_name) == 0) {
                eq_idx = i;
                break;
            }
        }
        if (eq_idx < 0) {
            std::fprintf(stderr, "Unknown equation: '%s'\nAvailable equations:\n", eq_name);
            for (const auto *e : equation_registry) {
                std::fprintf(stderr, "  - %s\n", e->name);
            }
        }
    }
    MPI_Bcast(&eq_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (eq_idx < 0) {
        MPI_Finalize();
        return 1;
    }
    const Equation *eq = equation_registry[eq_idx];

    // broadcast mesh filename
    int mesh_filename_len = (rank == 0) ? (int)std::strlen(mesh_filename) + 1 : 0;
    MPI_Bcast(&mesh_filename_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<char> mesh_filename_buf(mesh_filename_len);
    if (rank == 0) std::memcpy(mesh_filename_buf.data(), mesh_filename, mesh_filename_len);
    MPI_Bcast(mesh_filename_buf.data(), mesh_filename_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    uint64_t seed;
    if (rank == 0) seed = (uint64_t)std::time(nullptr);
    MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    prng_seed(seed ^ splitmix64((uint64_t)rank));

    // peek mesh dim, dispatch to equation's 2D or 3D entrypoint
    int dim;
    if (rank == 0) dim = peek_mesh_dim(mesh_filename_buf.data());
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rc;
    if (dim == 2) {
        rc = eq->run_2D(rank, size, mesh_filename_buf.data(), Nx, Ny, Nz, N_walks, epsilon);
    } else {
        rc = eq->run_3D(rank, size, mesh_filename_buf.data(), Nx, Ny, Nz, N_walks, epsilon);
    }

    MPI_Finalize();
    return rc;
}
