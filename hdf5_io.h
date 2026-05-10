#pragma once
#include <hdf5.h>
#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include "grid.h"

static inline void wos_write_hdf5(const char *filename,
                                  Grid grid,
                                  int y_start,
                                  int rank_Ny,
                                  const double (*u)[grid.Nx])
{
    // open hdf5
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    // setup dataset-transfer plist
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    // /u: each rank writes its section of the domain using hyperslab
    hsize_t u_dims[2] = { (hsize_t)grid.Ny, (hsize_t)grid.Nx };
    hid_t u_filespace = H5Screate_simple(2, u_dims, NULL);
    hid_t u_dset = H5Dcreate(file, "/u", H5T_NATIVE_DOUBLE, u_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t offset[2] = { (hsize_t)y_start, 0 };
    hsize_t count[2] = { (hsize_t)rank_Ny, (hsize_t)grid.Nx };
    H5Sselect_hyperslab(u_filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    hid_t u_memspace = H5Screate_simple(2, count, NULL);

    H5Dwrite(u_dset, H5T_NATIVE_DOUBLE, u_memspace, u_filespace, dxpl, u);

    H5Sclose(u_memspace);
    H5Sclose(u_filespace);

    // /x and /y coord arrays (same across all domains)
    double *x_arr = (double *)malloc((size_t)grid.Nx * sizeof *x_arr);
    double *y_arr = (double *)malloc((size_t)grid.Ny * sizeof *y_arr);
    for (int i = 0; i < grid.Nx; i++) {
        x_arr[i] = grid.xmin + (grid.xmax - grid.xmin) * (double)i / (grid.Nx - 1);
    }
    for (int i = 0; i < grid.Ny; i++) {
        y_arr[i] = grid.ymin + (grid.ymax - grid.ymin) * (double)i / (grid.Ny - 1);
    }

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

    free(x_arr);
    free(y_arr);

    H5Dclose(u_dset);
    H5Pclose(dxpl);
    H5Fclose(file);
}
