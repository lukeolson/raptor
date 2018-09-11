// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "raptor.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 5;
    int system = 0;
    double strong_threshold = 0.25;
    int iter;
    int num_variables = 1;

    ParMultilevel* ml;
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }
    if (system < 2)
    {
        int dim;
        double* stencil = NULL;
        aligned_vector<int> grid;
        if (argc > 2)
        {
            n = atoi(argv[2]);
        }

        if (system == 0)
        {
            dim = 3;
            grid.resize(dim, n);
            stencil = laplace_stencil_27pt();
        }
        else if (system == 1)
        {
            dim = 2;
            grid.resize(dim, n);
            double eps = 0.001;
            double theta = M_PI/4.0;
            if (argc > 3)
            {
                eps = atof(argv[3]);
                if (argc > 4)
                {
                    theta = atof(argv[4]);
                }
            }
            stencil = diffusion_stencil_2d(eps, theta);
        }
        A = par_stencil_grid(stencil, grid.data(), dim);
        delete[] stencil;
    }
#ifdef USING_MFEM
    else if (system == 2)
    {
        const char* mesh_file = argv[2];
        int mfem_system = 0;
        int order = 2;
        int seq_refines = 1;
        int par_refines = 1;
        int max_dofs = 1000000;
        if (argc > 3)
        {
            mfem_system = atoi(argv[3]);
            if (argc > 4)
            {
                order = atoi(argv[4]);
                if (argc > 5)
                {
                    seq_refines = atoi(argv[5]);
                    max_dofs = atoi(argv[5]);
                    if (argc > 6)
                    {
                        par_refines = atoi(argv[6]);
                    }
                }
            }
        }

        switch (mfem_system)
        {
            case 0:
                strong_threshold = 0.5;
                A = mfem_laplacian(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 1:
                strong_threshold = 0.0;
                A = mfem_grad_div(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 2:
                strong_threshold = 0.5;
                A = mfem_linear_elasticity(x, b, &num_variables, mesh_file, order, 
                        seq_refines, par_refines);
                break;
            case 3:
                A = mfem_adaptive_laplacian(x, b, mesh_file, order);
                x.set_const_value(1.0);
                A->mult(x, b);
                x.set_const_value(0.0);
                break;
            case 4:
                A = mfem_dg_diffusion(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 5:
                A = mfem_dg_elasticity(x, b, &num_variables, mesh_file, order, seq_refines, par_refines);
                break;
        }
    }
#endif
    else if (system == 3)
    {
        const char* file = "../../examples/LFAT5.pm";
        A = readParMatrix(file);
    }
    if (system != 2)
    {
        x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
        b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
        x.set_rand_values();
        A->mult(x, b);
        x.set_const_value(0.0);
    }

    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
            A->on_proc_column_map);

    int max_num_smooth = 3;

    // Warm Up
    ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation,
        Symmetric, SOR);
    ml->setup(A);
    ParVector tmp = ParVector(x);
    ml->solve(tmp, b);
    delete ml;

    // Smoothed Aggregation AMG
    if (rank == 0) printf("Standard AMG\n");
    for (int num_smooth = 1; num_smooth < max_num_smooth; num_smooth++)
    {
    	if (rank == 0) printf("\n\nNum Smooth Sweeps: %d:\n", num_smooth);
        ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation,
                Symmetric, SOR, num_smooth);
        ml->max_iterations = 1000;
        ml->solve_tol = 1e-05;
        t0 = MPI_Wtime();
        ml->setup(A);
        tfinal = MPI_Wtime() - t0;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Total Setup Time: %e\n", t0);
        ml->print_hierarchy();

        ParVector sas_sol = ParVector(x);
        t0 = MPI_Wtime();
        iter = ml->solve(sas_sol, b);
        tfinal = MPI_Wtime() - t0;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Total Solve Time: %e\n", t0);
        ml->print_residuals(iter);
        delete ml;
    }
       
    if (rank == 0) printf("Node-Aware:\n");
   // Warm Up
    ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation,
        Symmetric, SOR);
    ml->tap_amg = 3;
    ml->setup(A);
    ParVector nap_tmp = ParVector(x);
    ml->solve(nap_tmp, b);
    delete ml;

    // NAP Smoothed Aggregation AMG
    for (int num_smooth = 1; num_smooth < max_num_smooth; num_smooth++)
    {
    	if (rank == 0) printf("\n\nNum Smooth Sweeps: %d:\n", num_smooth);
        ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation,
                Symmetric, SOR, num_smooth);
        ml->max_iterations = 1000;
        ml->solve_tol = 1e-05;
        ml->tap_amg = 3;
        t0 = MPI_Wtime();
        ml->setup(A);
        tfinal = MPI_Wtime() - t0;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Total Setup Time: %e\n", t0);
        ml->print_hierarchy();

        ParVector sas_sol = ParVector(x);
        t0 = MPI_Wtime();
        iter = ml->solve(sas_sol, b);
        tfinal = MPI_Wtime() - t0;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Total Solve Time: %e\n", t0);
        ml->print_residuals(iter);
        delete ml;
    }
       

    delete A;

    MPI_Finalize();
    return 0;
}


