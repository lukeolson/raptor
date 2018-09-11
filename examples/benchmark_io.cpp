// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "core/types.hpp"
#include "gallery/par_matrix_IO.hpp"

//using namespace raptor;
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int dim=0;
    int n = 5;
    int system = 0;

    const char* file = "../../test_data/rss_A0.pm";
    if (argc > 1)
    {
        file = argv[1];
    }

    ParCSRMatrix* A=nullptr;
    ParVector x;
    ParVector b;

    double t0, tfinal;

    A = readParMatrix(file);
    x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
    b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    x.set_const_value(1.0);

    int n_spmvs = 100;
    int n_spmv_comms = 1000;
    int num_tests = 5;

    // Warm Up
    A->mult(x, b);
    A->comm->communicate(x);

    // Time SpMV (Multiple Tests)
    for (int i = 0; i < num_tests; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int j = 0; j < n_spmvs; j++)
        {
            A->mult(x, b);
        }
        tfinal = (MPI_Wtime() - t0) / n_spmvs;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Test %d Max SpMV Time: %e\n", i, t0);

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int j = 0; j < n_spmv_comms; j++)
        {
            A->comm->communicate(x);
        }
        tfinal = (MPI_Wtime() - t0) / n_spmv_comms;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Test %d Max SpMV Comm Time: %e\n", i, t0);
    }

    delete A;

    MPI_Finalize();

    return 0;
}

