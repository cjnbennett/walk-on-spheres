# Walk-on-Spheres PDE Solver
This repo implements a minimal walk-on-spheres-based PDE solver for various elliptic PDEs on arbitrary input 2D or 3D meshes.

## Dependencies
- C++17 compiler
- CMake >= 3.20
- MPI (e.g. Open MPI, MPICH, Cray MPI)
- HDF5 with parallel support
- Python >= 3.10 with `numpy`, `h5py`, `matplotlib` (visualisation only)

### MacOS
```bash
brew install cmake open-mpi hdf5-mpi
```

### HPC
Load the appropriate modules, for example:
```bash
module load cmake cray-hdf5-parallel
```

Submit using a slurm script.

## Build
```bash
cmake -B build
cmake --build build -j
```

## Run
```bash
mpirun -np <ranks> ./build/wos [equation] [Nx Ny] [Nz] [mesh.obj]
```

All arguments are positional and optional. The dimension of the mesh is auto-detected from the file. Pass an unknown equation name to get the list of registered equations.

Examples:

```bash
# 2D screened Poisson on the annulus, 64x64 grid, 8 ranks
mpirun -np 8 ./build/wos screened_poisson 64 64 meshes/annulus.obj

# 3D screened Poisson on the sphere, 32x32x32 grid
mpirun -np 8 ./build/wos screened_poisson 32 32 32 meshes/sphere.obj
```

Output is written to `wos.h5` in the current directory.

## Visualise
```bash
pip install -r requirements.txt
python plot.py --mesh meshes/annulus.obj wos.h5
```

The `--mesh` argument must match the mesh used to produce the `.h5` file. The output file is `u.png` by default.

## Adding a new equation
Each equation lives in its own `src/equations/<name>.cpp` and is registered in `src/equations/equations.hpp`. When creating a new equation, the relevant files need to be added to the CMakeLists.txt.
