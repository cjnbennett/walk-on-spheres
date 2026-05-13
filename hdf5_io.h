#pragma once
#include <hdf5.h>
#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include "grid.h"

// write the WoS solution u(x,y,z) to HDF5, each rank contributing its assigned sub-block
static inline void wos_write_hdf5(const char *filename,
                                  Grid grid,
                                  int x_start, int y_start, int z_start,
                                  int block_Nx, int block_Ny, int block_Nz,
                                  const double *u)
{
    // open hdf5
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    // setup dataset-transfer plist
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    // /u: global shape (Nx, Ny, Nz); each rank writes its (block_Nx, block_Ny, block_Nz) hyperslab
    hsize_t u_dims[3] = { (hsize_t)grid.Nx, (hsize_t)grid.Ny, (hsize_t)grid.Nz };
    hid_t u_filespace = H5Screate_simple(3, u_dims, NULL);
    hid_t u_dset = H5Dcreate(file, "/u", H5T_NATIVE_DOUBLE, u_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t offset[3] = { (hsize_t)x_start, (hsize_t)y_start, (hsize_t)z_start };
    hsize_t count[3] = { (hsize_t)block_Nx, (hsize_t)block_Ny, (hsize_t)block_Nz };
    H5Sselect_hyperslab(u_filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    hid_t u_memspace = H5Screate_simple(3, count, NULL);

    H5Dwrite(u_dset, H5T_NATIVE_DOUBLE, u_memspace, u_filespace, dxpl, u);

    H5Sclose(u_memspace);
    H5Sclose(u_filespace);

    // /x, /y, /z coord arrays (same across all ranks)
    double *x_arr = (double *)malloc((size_t)grid.Nx * sizeof *x_arr);
    double *y_arr = (double *)malloc((size_t)grid.Ny * sizeof *y_arr);
    double *z_arr = (double *)malloc((size_t)grid.Nz * sizeof *z_arr);
    for (int i = 0; i < grid.Nx; i++)
        x_arr[i] = (grid.Nx > 1) ? grid.xmin + (grid.xmax - grid.xmin) * (double)i / (grid.Nx - 1) : grid.xmin;
    for (int i = 0; i < grid.Ny; i++)
        y_arr[i] = (grid.Ny > 1) ? grid.ymin + (grid.ymax - grid.ymin) * (double)i / (grid.Ny - 1) : grid.ymin;
    for (int i = 0; i < grid.Nz; i++)
        z_arr[i] = (grid.Nz > 1) ? grid.zmin + (grid.zmax - grid.zmin) * (double)i / (grid.Nz - 1) : grid.zmin;

    hsize_t x_dims[1] = { (hsize_t)grid.Nx };
    hid_t x_space = H5Screate_simple(1, x_dims, NULL);
    hid_t x_dset = H5Dcreate(file, "/x", H5T_NATIVE_DOUBLE, x_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(x_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl, x_arr);
    H5Dclose(x_dset);
    H5Sclose(x_space);

    hsize_t y_dims[1] = { (hsize_t)grid.Ny };
    hid_t y_space = H5Screate_simple(1, y_dims, NULL);
    hid_t y_dset = H5Dcreate(file, "/y", H5T_NATIVE_DOUBLE, y_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(y_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl, y_arr);
    H5Dclose(y_dset);
    H5Sclose(y_space);

    hsize_t z_dims[1] = { (hsize_t)grid.Nz };
    hid_t z_space = H5Screate_simple(1, z_dims, NULL);
    hid_t z_dset = H5Dcreate(file, "/z", H5T_NATIVE_DOUBLE, z_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(z_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl, z_arr);
    H5Dclose(z_dset);
    H5Sclose(z_space);

    free(x_arr);
    free(y_arr);
    free(z_arr);

    H5Dclose(u_dset);
    H5Pclose(dxpl);
    H5Fclose(file);
}
