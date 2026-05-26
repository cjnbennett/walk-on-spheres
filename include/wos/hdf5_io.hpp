#pragma once
#include <hdf5.h>
#include <mpi.h>
#include <vector>
#include "wos/field_3D.hpp"
#include "wos/grid.hpp"

namespace wos {

// write the WoS solution u(x,y,z) to HDF5, each rank contributing its assigned sub-block
inline void wos_write_hdf5(const char *filename,
                           Grid grid,
                           int x_start, int y_start, int z_start,
                           const Field3D &u)
{
    // open hdf5
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    // setup dataset-transfer plist
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    // /u: global shape (Nx, Ny, Nz); each rank writes its (u.Nx, u.Ny, u.Nz) hyperslab
    hsize_t u_dims[3] = { (hsize_t)grid.Nx, (hsize_t)grid.Ny, (hsize_t)grid.Nz };
    hid_t u_filespace = H5Screate_simple(3, u_dims, nullptr);
    hid_t u_dset = H5Dcreate(file, "/u", H5T_NATIVE_DOUBLE, u_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t offset[3] = { (hsize_t)x_start, (hsize_t)y_start, (hsize_t)z_start };
    hsize_t count[3]  = { (hsize_t)u.Nx,    (hsize_t)u.Ny,    (hsize_t)u.Nz };
    H5Sselect_hyperslab(u_filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
    hid_t u_memspace = H5Screate_simple(3, count, nullptr);

    H5Dwrite(u_dset, H5T_NATIVE_DOUBLE, u_memspace, u_filespace, dxpl, u.data.data());

    H5Sclose(u_memspace);
    H5Sclose(u_filespace);

    // /x, /y, /z coord arrays (same across all ranks)
    std::vector<double> x_arr(grid.Nx), y_arr(grid.Ny), z_arr(grid.Nz);
    for (int i = 0; i < grid.Nx; i++)
        x_arr[i] = (grid.Nx > 1) ? grid.xmin + (grid.xmax - grid.xmin) * (double)i / (grid.Nx - 1) : grid.xmin;
    for (int i = 0; i < grid.Ny; i++)
        y_arr[i] = (grid.Ny > 1) ? grid.ymin + (grid.ymax - grid.ymin) * (double)i / (grid.Ny - 1) : grid.ymin;
    for (int i = 0; i < grid.Nz; i++)
        z_arr[i] = (grid.Nz > 1) ? grid.zmin + (grid.zmax - grid.zmin) * (double)i / (grid.Nz - 1) : grid.zmin;

    auto write_1d = [&](const char *name, hsize_t n, const double *buf) {
        hid_t space = H5Screate_simple(1, &n, nullptr);
        hid_t dset = H5Dcreate(file, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl, buf);
        H5Dclose(dset);
        H5Sclose(space);
    };
    write_1d("/x", grid.Nx, x_arr.data());
    write_1d("/y", grid.Ny, y_arr.data());
    write_1d("/z", grid.Nz, z_arr.data());

    H5Dclose(u_dset);
    H5Pclose(dxpl);
    H5Fclose(file);
}

}
