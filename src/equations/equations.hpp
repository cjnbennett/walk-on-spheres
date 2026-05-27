#pragma once

struct Equation {
    const char *name;
    int (*run_2D)(int rank, int size, const char *mesh_filename, int Nx, int Ny, int Nz, int N_walks, double epsilon);
    int (*run_3D)(int rank, int size, const char *mesh_filename, int Nx, int Ny, int Nz, int N_walks, double epsilon);
};

// link equations
extern const Equation laplace;
extern const Equation poisson;
extern const Equation screened_poisson;
extern const Equation helmholtz;

// equation registry
inline const Equation *const equation_registry[] = {
    &laplace,
    &poisson,
    &screened_poisson,
    &helmholtz,
};
