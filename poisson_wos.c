// This file implements a simple walk-on-spheres (WoS) to solve the 2D Poisson equation ∆u(x,y) = -f(x,y)
// on a domain Ω, with boundary condition u(x,y) = g(x,y) on ∂Ω. Here Ω is an arbitary 2D mesh. To
// recover the 2D Laplace equation, set f(x,y) = 0.
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "grid.h"
#include "hash.h"
#include "hdf5_io.h"
#include "mesh.h"
#include "inside.h"
#include "npq.h"

// 2D sphere for WoS
typedef struct {
    Point2D centre;
    double radius;
} Sphere;

// f: source function, defined on Ω
double f(Point2D p) {
    // a bump centred at (0.65, 0) with radius 0.15
    return sqrt(pow(p.x - 0.65, 2) + pow(p.y, 2)) <= 0.15 ? 10.0 : 0.0;
}

// g: boundary value function for u(x,y), defined for (x,y) ∈ ∂Ω
double g(Point2D p) {
    double r = sqrt(p.x*p.x + p.y*p.y);
    return r > 0.7 ? p.x*p.x - p.y*p.y      // outer: saddle
                   : 0.0;                   // inner: zero
}

// Green's function for PDE operator on spherical domain
double G(Sphere sphere, Point2D x, Point2D y) {
    // Green's function for Laplace operator on 2D spherical domain ("harmonic")
    return log(sphere.radius / dist_2D(x,y)) / (2*M_PI);
}

// perform a singular step of the walk exiting on a sphere
Point2D step(Sphere sphere) {
    // for standard WoS, the exit distribution is uniform
    double theta = (double)rand() / (double)RAND_MAX * 2*M_PI;
    Point2D exit = {
        .x = sphere.centre.x + sphere.radius * cos(theta),
        .y = sphere.centre.y + sphere.radius * sin(theta)
    };
    return exit;
}

// draw a sample uniformly from a sphere, e.g. for sampling the source contribution
Point2D source_sample(Sphere sphere) {
    // uniform sampling
    double theta = (double)rand() / (double)RAND_MAX * 2*M_PI;
    double r = sphere.radius * sqrt((double)rand() / (double)RAND_MAX);
    Point2D sp = {
        .x = sphere.centre.x + r * cos(theta),
        .y = sphere.centre.y + r * sin(theta)
    };
    return sp;
}

// perform Monte Carlo walk-on-spheres to estimate u(p0) at a point p0
double wos(const Mesh2D *Ω, Point2D p0, int N_walks, double epsilon) {
    double mean = 0.0;

    for (int i = 0; i < N_walks; i++) {
        Point2D p = p0;
        Point2D np;
        double r = npq_mesh_2D(Ω, p, &np);
        Sphere sphere = {
            .centre = p,
            .radius = r
        };

        double sample = 0.0;

        while (sphere.radius > epsilon) {
            // for a PDE containing a source term (e.g. Poisson equation), we must accumulate a source
            // contribution for non-exiting walks
            Point2D sp = source_sample(sphere);
            sample += M_PI*pow(sphere.radius,2) * f(sp) * G(sphere, p, sp);
            
            // iterate
            p = step(sphere);
            r = npq_mesh_2D(Ω, p, &np);
            sphere = (Sphere){
                .centre = p,
                .radius = r
            };
        }

        sample += g(np);
        mean += (sample - mean) / (i+1);    // Welford mean
    }

    return mean;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *mesh_filename = NULL; // path to domain mesh .obj on disk
    int Nx = 128, Ny = 128;     // number of points sampled on domain
    int N_walks = 1e5;          // number of walks simulated per point
    double epsilon = 1e-6;      // tolerance for WoS to boundary

    // rank 0 determines parameters for all ranks
    if (rank == 0) {
        if (argc >= 3) {
            Nx = atoi(argv[1]);
            Ny = atoi(argv[2]);
        }
        if (argc >= 4) {
            mesh_filename = argv[3];
        } else {
            mesh_filename = "meshes/annulus.obj";
        }
    }

    // guard against too many nodes being launched
    if (rank == 0 && size > Ny) {
        fprintf(stderr, "too many nodes launched, size %d > Ny %d\n", size, Ny);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Bcast(&Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_walks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    uint64_t seed;
    if (rank == 0) seed = (uint64_t)time(NULL);
    MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    srand((unsigned int)(seed ^ splitmix64((uint64_t)rank)));   // note: this is probably a really bad way to set the seed

    // load mesh as domain
    Mesh2D Ω = {0};
    if (rank == 0) {
        printf("Reading mesh from disk (%s)...\n", mesh_filename);
        Ω = load_mesh_2D(mesh_filename);
        printf("Finished reading. Broadcasting mesh...\n");
    }
    bcast_mesh_2D(&Ω, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Finished broadcasting.\n");

    // setup domain discretisation
    Grid grid = {
        .Nx = Nx,
        .Ny = Ny,
        .xmin = -1.0,   // TODO - implement bounding box query for Mesh2D
        .xmax = 1.0,
        .ymin = -1.0,
        .ymax = 1.0
    };

    // divide the domain per rank
    int y_start = (grid.Ny * rank) / size;
    int y_end = (grid.Ny * (rank + 1)) / size;
    int rank_Ny = y_end - y_start;

    double (*u)[grid.Nx] = malloc((size_t)rank_Ny * grid.Nx * sizeof *u);
    if (!u) {
        fprintf(stderr, "malloc failed for rank %d, aborting\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) printf("Beginning walk-on-spheres...\n");

    // run WoS over rank's subsection of domain
    for (int i = 0; i < rank_Ny; i++) {
        double y = grid.ymin + (grid.ymax - grid.ymin) * (y_start + i) / (grid.Ny - 1);
        for (int j = 0; j < grid.Nx; j++) {
            double x = grid.xmin + (grid.xmax - grid.xmin) * j / (grid.Nx - 1);
            Point2D p0 = { x, y };
            // sample only points in the domain
            u[i][j] = (inside_mesh_2D(&Ω, p0) ? wos(&Ω, p0, N_walks, epsilon) : NAN);
        }
    }

    free_mesh_2D(&Ω);
    // middle rank (most likely to have "most work" to do) should broadcast progress update
    if (rank == size/2) printf("Finished walk-on-spheres. Writing results...\n");

    // save to hdf5
    wos_write_hdf5("poisson_wos.h5", grid, y_start, rank_Ny, u);
    if (rank == 0) printf("Finished writing.\n");

    free(u);
    MPI_Finalize();
    return 0;
}
