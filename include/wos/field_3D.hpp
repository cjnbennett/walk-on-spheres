#pragma once
#include <cstddef>
#include <vector>

namespace wos {

// 3D box field, row-major (i, j, k). Single contiguous allocation for hdf5 write.
struct Field3D {
    int Nx, Ny, Nz;
    std::vector<double> data;

    Field3D(int Nx, int Ny, int Nz)
        : Nx(Nx), Ny(Ny), Nz(Nz), data(static_cast<std::size_t>(Nx) * Ny * Nz) {}

    double &operator()(int i, int j, int k) {
        return data[flat(i, j, k)];
    }
    double  operator()(int i, int j, int k) const {
        return data[flat(i, j, k)];
    }

private:
    std::size_t flat(int i, int j, int k) const {
        return (static_cast<std::size_t>(i) * Ny + j) * Nz + k;
    }
};

}
