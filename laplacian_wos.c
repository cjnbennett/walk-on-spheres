// This file implements a simple walk-on-spheres (WoS) to solve the 2D Laplace equation ∆u(x,y) = 0 on a domain
// Ω, with boundary condition u(x,y) = g(x,y) on ∂Ω. Here Ω is the unit circle, and g(x,y) a paraboloid
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

// nearest point query
Point npq(CircleBoundary boundary, Point from) {
    double angle = atan2(from.y - boundary.centre.y, from.x - boundary.centre.x);
    Point np = {
        .x = boundary.centre.x + boundary.radius * cos(angle),
        .y = boundary.centre.y + boundary.radius * sin(angle)
    };
    return np;
}

// for the Laplacian, we sample uniformly from the constructed sphere
Point walk(Sphere sphere) {
    double theta = (double)rand() / (double)RAND_MAX * 2*M_PI;
    Point exit = {
        .x = sphere.centre.x + sphere.radius * cos(theta),
        .y = sphere.centre.y + sphere.radius * sin(theta)
    };
    return exit;
}

double dist(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// g: boundary value function for u(x,y), defined for (x,y) ∈ ∂Ω
double g(Point p) {
    return pow(p.x, 2) - pow(p.y, 2);     // hyperbolic paraboloid (saddle)
}

int main()
{
    int N = 1e5;                    // number of samples
    double epsilon = 1e-6;          // tolerance for WoS to boundary
    srand((unsigned)time(NULL));    // TODO ensure different per MPI process (e.g. xor with hash of rank)

    // define our domain boundary
    Point origin = { .x = 0.0, .y = 0.0 };
    CircleBoundary boundary = {
        .centre = origin,
        .radius = 1.0
    };

    // point to sample
    Point p0 = { .x = 0.358, .y = 0.134 }; // can embarrassingly parallelise this to many points and reconstruct the entire function u!
    double mean = 0.0;   // best estimate of u(p0)

    for (int i = 0; i < N; i++) {
        Point p = p0;
        Point np = npq(boundary, p);
        Sphere sphere = {
            .centre = p,
            .radius = dist(np, p)
        };

        while (sphere.radius > epsilon) {
            p = walk(sphere);
            np = npq(boundary, p);
            sphere = (Sphere){
                .centre = p,
                .radius = dist(np, p)
            };
        }

        double sample = g(np);
        mean += (sample - mean) / (i+1);    // Welford mean
    }

    printf("Simulated: u(%f,%f) = %f\n", p0.x, p0.y, mean);

    return 0;
}
