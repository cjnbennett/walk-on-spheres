// This file implements a simple walk-on-spheres (WoS) to solve the 2D Laplace equation ∆u(x,y) = 0 on a
// domain Ω, with boundary condition u(x,y) = g(x,y) on ∂Ω. Here Ω is the unit circle, and g(x,y) a
// paraboloid.
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

typedef struct { double x, y; } Point;

typedef struct {
    Point centre;
    double radius;
} CircleBoundary;

// sphere for WoS - semantically different from domain boundary (CircleBoundary)
typedef struct {
    Point centre;
    double radius;
} Sphere;

// g: boundary value function for u(x,y), defined for (x,y) ∈ ∂Ω
double g(Point p) {
    return pow(p.x, 2) - pow(p.y, 2);     // hyperbolic paraboloid (saddle)
}

// check whether a point is in the domain
bool in_domain(CircleBoundary boundary, Point p) {
    return pow(p.x - boundary.centre.x, 2) + pow(p.y - boundary.centre.y, 2) <= pow(boundary.radius,2);
}

// nearest point query - return nearest point on the boundary to a given point
Point npq(CircleBoundary boundary, Point from) {
    double angle = atan2(from.y - boundary.centre.y, from.x - boundary.centre.x);
    Point np = {
        .x = boundary.centre.x + boundary.radius * cos(angle),
        .y = boundary.centre.y + boundary.radius * sin(angle)
    };
    return np;
}

double dist(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// perform a singular step of the walk exiting on a sphere
Point step(Sphere sphere) {
    // for the Laplacian, the exit distribution is uniform
    double theta = (double)rand() / (double)RAND_MAX * 2*M_PI;
    Point exit = {
        .x = sphere.centre.x + sphere.radius * cos(theta),
        .y = sphere.centre.y + sphere.radius * sin(theta)
    };
    return exit;
}

// perform Monte Carlo walk-on-spheres to estimate u(p0) at a point p0
double wos(CircleBoundary boundary, Point p0, int N_walks, double epsilon) {
    double mean = 0.0;

    for (int i = 0; i < N_walks; i++) {
        Point p = p0;
        Point np = npq(boundary, p);
        Sphere sphere = {
            .centre = p,
            .radius = dist(np, p)
        };

        while (sphere.radius > epsilon) {
            p = step(sphere);
            np = npq(boundary, p);
            sphere = (Sphere){
                .centre = p,
                .radius = dist(np, p)
            };
        }

        double sample = g(np);
        mean += (sample - mean) / (i+1);    // Welford mean
    }

    return mean;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int Nx = 128, Ny = 128;     // number of points sampled on domain
    int N_walks = 1e5;          // number of walks simulated per point
    double epsilon = 1e-6;      // tolerance for WoS to boundary

    // rank 0 determines parameters for all ranks
    if (rank == 0) {
        if (argc >= 3) {
            Nx = atoi(argv[1]);
            Ny = atoi(argv[2]);
        }
    }

    MPI_Bcast(&Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_walks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    uint64_t seed;
    if (rank == 0) seed = (uint64_t)time(NULL);
    MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    srand((unsigned int)(seed ^ splitmix64((uint64_t)rank)));   // note: this is probably a really bad way to set the seed

    // define our domain boundary
    Point origin = { .x = 0.0, .y = 0.0 };
    CircleBoundary boundary = {
        .centre = origin,
        .radius = 1.0
    };

    // setup domain discretisation
    Grid grid = {
        .Nx = Nx,
        .Ny = Ny,
        .xmin = -1.0,
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

    // run WoS over rank's subsection of domain
    for (int i = 0; i < rank_Ny; i++) {
        double y = grid.ymin + (grid.ymax - grid.ymin) * (y_start + i) / (grid.Ny - 1);
        for (int j = 0; j < grid.Nx; j++) {
            double x = grid.xmin + (grid.xmax - grid.xmin) * j / (grid.Nx - 1);
            Point p0 = { x, y };
            // sample only points in the domain
            u[i][j] = (in_domain(boundary, p0) ? wos(boundary, p0, N_walks, epsilon) : NAN);
        }
    }

    // save to hdf5
    wos_write_hdf5("laplacian_wos.h5", grid, y_start, rank_Ny, u);

    free(u);
    MPI_Finalize();
    return 0;
}
