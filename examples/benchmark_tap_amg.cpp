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

    coarsen_t coarsen_type = HMIS;
    interp_t interp_type = Extended;

    int agg_num_levels = 0;
    int p_max_elmts = 0;
    int h_interp_type = 6;
    int h_coarsen_type = 10;

    int tap_rs = 3;
    int tap_sa = 3;

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
            coarsen_type = Falgout;
            interp_type = ModClassical;
            tap_rs = 5;

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

        coarsen_type = HMIS;
        interp_type = Extended;
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


    // Convert system to Hypre format 
    HYPRE_IJMatrix A_h_ij = convert(A);
    HYPRE_IJVector x_h_ij = convert(x);
    HYPRE_IJVector b_h_ij = convert(b);
    hypre_ParCSRMatrix* A_h;
    HYPRE_IJMatrixGetObject(A_h_ij, (void**) &A_h);
    hypre_ParVector* x_h;
    HYPRE_IJVectorGetObject(x_h_ij, (void **) &x_h);
    hypre_ParVector* b_h;
    HYPRE_IJVectorGetObject(b_h_ij, (void **) &b_h);

    HYPRE_Solver solver_data;

    double* x_h_data = hypre_VectorData(hypre_ParVectorLocalVector(x_h));
    for (int i = 0; i < A->local_num_rows; i++)
    {
        x_h_data[i] = x[i];
    }

    // Create Hypre Hierarchy
    double hypre_setup, hypre_solve;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    solver_data = hypre_create_hierarchy(A_h, x_h, b_h,
                            h_coarsen_type, h_interp_type, p_max_elmts, agg_num_levels,
                            strong_threshold);
    hypre_setup = MPI_Wtime() - t0;

    // Solve Hypre Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    HYPRE_BoomerAMGSolve(solver_data, A_h, b_h, x_h);
    hypre_solve = MPI_Wtime() - t0;

    MPI_Reduce(&hypre_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Hypre Setup: %e\n", t0);
    MPI_Reduce(&hypre_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Hypre Solve: %e\n", t0);

    // Delete hypre hierarchy
    hypre_BoomerAMGDestroy(solver_data);

    // Ruge-Stuben AMG
    if (rank == 0) printf("Ruge Stuben Solver: \n");
    MPI_Barrier(MPI_COMM_WORLD);
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, SOR);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-05;
    ml->num_variables = num_variables;
    //ml->track_times = true;
    t0 = MPI_Wtime();
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    ml->print_hierarchy();
    //ml->print_setup_times();

    MPI_Barrier(MPI_COMM_WORLD);
    ParVector rss_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(rss_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    ml->print_residuals(iter);
    //ml->print_solve_times();
    delete ml;

    // TAP Ruge-Stuben AMG
    if (rank == 0) printf("\n\nTAP Ruge Stuben Solver: \n");
    MPI_Barrier(MPI_COMM_WORLD);
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, SOR);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-05;
    ml->num_variables = num_variables;
    //ml->track_times = true;
    ml->tap_amg = tap_rs;
    t0 = MPI_Wtime();
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    ml->print_hierarchy();
    //ml->print_setup_times();

    MPI_Barrier(MPI_COMM_WORLD);
    ParVector tap_rss_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(tap_rss_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    ml->print_residuals(iter);
    //ml->print_solve_times();
    delete ml;

    // Smoothed Aggregation AMG
    if (rank == 0) printf("\n\nSmoothed Aggregation Solver:\n");
    ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation, 
            Symmetric, SOR);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-05;
    //ml->track_times = true;
    t0 = MPI_Wtime();
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    ml->print_hierarchy();
    //ml->print_setup_times();

    ParVector sas_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(sas_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    ml->print_residuals(iter);
    //ml->print_solve_times();
    delete ml;

    // TAPSmoothed Aggregation AMG
    if (rank == 0) printf("\n\nTAP Smoothed Aggregation Solver:\n");
    ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation,
            Symmetric, SOR);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-05;
    //ml->track_times = true;
    ml->tap_amg = tap_sa;
    t0 = MPI_Wtime();
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    ml->print_hierarchy();
    //ml->print_setup_times();

    ParVector tap_sas_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(tap_sas_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    ml->print_residuals(iter);
    //ml->print_solve_times();
    delete ml;





    HYPRE_IJMatrixDestroy(A_h_ij);
    HYPRE_IJVectorDestroy(x_h_ij);
    HYPRE_IJVectorDestroy(b_h_ij);

    delete A;

    MPI_Finalize();
    return 0;
}


