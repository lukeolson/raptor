// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "core/types.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "multilevel/par_multilevel.hpp"
#include "ruge_stuben/par_ruge_stuben_solver.hpp"

#ifdef USING_MFEM
  #include "gallery/external/mfem_wrapper.hpp"
#endif

#define eager_cutoff 1000
#define short_cutoff 62

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim;
    int n = 5;
    int system = 0;

    // Cube topology info
    int num_nodes = num_procs / 16;
    int node = rank / 16;
    int num_dir = cbrt(num_nodes);
    if (num_dir * num_dir * num_dir < num_nodes)
        num_dir++;

    int x_dim = num_dir;
    int y_dim = x_dim * num_dir;
    int z_dim = y_dim * num_dir;

    int x_pos = node % num_dir;
    int y_pos = (node / x_dim) % num_dir;
    int z_pos = (node / y_dim) % num_dir;

    aligned_vector<int> my_cube_pos(3);
    my_cube_pos[0] = x_pos;
    my_cube_pos[1] = y_pos;
    my_cube_pos[2] = z_pos;
    aligned_vector<int> cube_pos(3*num_procs);
    MPI_Allgather(my_cube_pos.data(), 3, MPI_INT, cube_pos.data(), 3, MPI_INT, MPI_COMM_WORLD);
 
    aligned_vector<int> proc_distances(num_procs);
    for (int i = 0; i < num_procs; i++)
    {
        proc_distances[i] = (fabs(cube_pos[i*3] - x_pos) + fabs(cube_pos[i*3+1] - y_pos)
                + fabs(cube_pos[i*3+2] - z_pos));
    }

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;
    double t0_comm, tfinal_comm;
    int n0, s0;
    int nfinal, sfinal;
    double raptor_setup, raptor_solve;
    int num_variables = 1;
    relax_t relax_type = SOR;
    coarsen_t coarsen_type = CLJP;
    interp_t interp_type = ModClassical;
    double strong_threshold = 0.25;

    aligned_vector<double> residuals;

    if (system < 2)
    {
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
            strong_threshold = 0.0;
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
        strong_threshold = 0.0;
        switch (mfem_system)
        {
            case 0:
                A = mfem_laplacian(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 1:
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
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
                A->on_proc_column_map);
        x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
        b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
        x.set_rand_values();
        A->mult(x, b);
    }

    ParMultilevel* ml;

    // Setup Raptor Hierarchy
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, relax_type);
    ml->num_variables = num_variables;
    ml->setup(A);

    int n_tests = 100;
    int n_spmv_tests = 10000;
    int active, sum_active;
    CSRMatrix* C;
    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParCSRMatrix* Pl = ml->levels[i]->P;
        ParCSRMatrix* Sl = Al->strength(Classical, strong_threshold);
        if (!Al->comm) Al->comm = new ParComm(Al->partition, Al->off_proc_column_map, 
                Al->on_proc_column_map);
        if (!Al->tap_comm) Al->tap_comm = new TAPComm(Al->partition,
                Al->off_proc_column_map, Al->on_proc_column_map);
        if (!Sl->comm) Sl->comm = new ParComm(Sl->partition, Sl->off_proc_column_map, 
                Sl->on_proc_column_map);
        if (!Sl->tap_comm) Sl->tap_comm = new TAPComm(Sl->partition,
                Sl->off_proc_column_map, Sl->on_proc_column_map);
        if (!Pl->comm) Pl->comm = new ParComm(Pl->partition, Pl->off_proc_column_map, 
                Pl->on_proc_column_map);
        if (!Pl->tap_comm) Pl->tap_comm = new TAPComm(Pl->partition,
                Pl->off_proc_column_map, Pl->on_proc_column_map);

        aligned_vector<int> rowptr_S;
        aligned_vector<int> cols_S;
        aligned_vector<double> vals_S;
        rowptr_S.push_back(0);
        for (int i = 0; i < Sl->local_num_rows; i++)
        {
            for (int j = Sl->on_proc->idx1[i]; j < Sl->on_proc->idx1[i+1]; j++)
            {
                cols_S.push_back(Sl->on_proc->idx2[j]);
                vals_S.push_back(1.0);
            }
            for (int j = Sl->off_proc->idx1[i]; j < Sl->off_proc->idx1[i+1]; j++)
            {
                cols_S.push_back(Sl->off_proc->idx2[j]);
                vals_S.push_back(1.0);
            }
            rowptr_S.push_back(cols_S.size());
        }



        if (rank == 0) printf("Level %d\n", i);

        // Matrix Communication with A
        if (rank == 0) printf("A Communication:\n");
        Al->print_mult(Al, proc_distances);
        active = 1;
        if (Al->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active); 
        C = Al->comm->communicate(Al);
        delete C;
        tfinal = 0;
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            C = Al->comm->communicate(Al);
            tfinal += (MPI_Wtime() - t0);
            delete C;
        }
        tfinal /= n_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);

        // Matrix Communication with S
        if (rank == 0) printf("S Communication:\n");
        Sl->print_mult(Sl, proc_distances);
        active = 1;
        if (Sl->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active);
        C = Sl->comm->communicate(rowptr_S, cols_S, vals_S);
        delete C;
        tfinal = 0;
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            C = Sl->comm->communicate(rowptr_S, cols_S, vals_S);
            tfinal += (MPI_Wtime() - t0);
            delete C;
        }
        tfinal /= n_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);

        // Matrix Communication with P
        if (rank == 0) printf("P Communication:\n");
        Pl->print_mult(Pl, proc_distances);
        active = 1;
        if (Pl->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active); 
        C = Pl->comm->communicate(Pl);
        delete C;
        tfinal = 0;
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            C = Pl->comm->communicate(Pl);
            tfinal += (MPI_Wtime() - t0);
            delete C;
        }
        tfinal /= n_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);


        // Matrix Communication with A
        if (rank == 0) printf("A TAP Communication:\n");
        Al->print_tap_mult(Al, proc_distances);
        active = 0;
        if (Al->tap_comm->local_L_par_comm->send_data->num_msgs || 
            Al->tap_comm->local_S_par_comm->send_data->num_msgs || 
            Al->tap_comm->local_R_par_comm->send_data->num_msgs || 
            Al->tap_comm->global_par_comm->send_data->num_msgs)
               active = 1;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active); 
        C = Al->tap_comm->communicate(Al);
        delete C;
        tfinal = 0;
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            C = Al->tap_comm->communicate(Al);
            tfinal += (MPI_Wtime() - t0);
            delete C;
        }
        tfinal /= n_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);

        // Matrix Communication with S 
        if (rank == 0) printf("S TAP Communication:\n");
        Sl->print_tap_mult(Sl, proc_distances);
        active = 0;
        if (Sl->tap_comm->local_L_par_comm->send_data->num_msgs || 
            Sl->tap_comm->local_S_par_comm->send_data->num_msgs || 
            Sl->tap_comm->local_R_par_comm->send_data->num_msgs || 
            Sl->tap_comm->global_par_comm->send_data->num_msgs)
               active = 1;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active); 
        C = Sl->tap_comm->communicate(rowptr_S, cols_S, vals_S);
        delete C;
        tfinal = 0;
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            C = Sl->tap_comm->communicate(rowptr_S, cols_S, vals_S);
            tfinal += (MPI_Wtime() - t0);
            delete C;
        }
        tfinal /= n_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);

        // Matrix Communication with P
        if (rank == 0) printf("P TAP Communication:\n");
        Pl->print_tap_mult(Pl, proc_distances);
        active = 0;
        if (Pl->tap_comm->local_L_par_comm->send_data->num_msgs || 
            Pl->tap_comm->local_S_par_comm->send_data->num_msgs || 
            Pl->tap_comm->local_R_par_comm->send_data->num_msgs || 
            Pl->tap_comm->global_par_comm->send_data->num_msgs)
               active = 1;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active); 
        C = Pl->tap_comm->communicate(Pl);
        delete C;
        tfinal = 0;
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            C = Pl->tap_comm->communicate(Pl);
            tfinal += (MPI_Wtime() - t0);
            delete C;
        }
        tfinal /= n_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);




        // Vector communication with A
        MPI_Barrier(MPI_COMM_WORLD);
        ml->levels[i]->x.set_const_value(0.0);
        if (rank == 0) printf("A Vector Communication\n");
	Al->print_mult(proc_distances);
        active = 1;
        if (Al->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active);
        Al->comm->communicate(ml->levels[i]->x);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_spmv_tests; test++)
        {
            // b <- A*x
            Al->comm->communicate(ml->levels[i]->x);
        }
        tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);

        // Vector communication with S
        MPI_Barrier(MPI_COMM_WORLD);
        ml->levels[i]->x.set_const_value(0.0);
        if (rank == 0) printf("S Vector Communication\n");
        Sl->print_mult(proc_distances);
        active = 1;
        if (Sl->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active);
        Sl->comm->communicate(ml->levels[i]->x);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_spmv_tests; test++)
        {
            // b <- A*x
            Sl->comm->communicate(ml->levels[i]->x);
        }
        tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);

        // Vector communication with P
        MPI_Barrier(MPI_COMM_WORLD);
        ml->levels[i+1]->x.set_const_value(0.0);
        if (rank == 0) printf("P Vector Communication\n");
        Pl->print_mult(proc_distances);
        active = 1;
        if (Pl->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active);
        Pl->comm->communicate(ml->levels[i+1]->x);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_spmv_tests; test++)
        {
            // b <- A*x
            Pl->comm->communicate(ml->levels[i+1]->x);
        }
        tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);


        // Vector communication with A
        MPI_Barrier(MPI_COMM_WORLD);
        ml->levels[i]->x.set_const_value(0.0);
        if (rank == 0) printf("A Vector TAP Communication\n");
        Al->print_tap_mult(proc_distances);
        active = 1;
        if (Al->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active);
        Al->tap_comm->communicate(ml->levels[i]->x);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_spmv_tests; test++)
        {
            // b <- A*x
            Al->tap_comm->communicate(ml->levels[i]->x);
        }
        tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);

        // Vector communication with S
        MPI_Barrier(MPI_COMM_WORLD);
        ml->levels[i]->x.set_const_value(0.0);
        if (rank == 0) printf("S Vector TAP Communication\n");
        Sl->print_tap_mult(proc_distances);
        active = 1;
        if (Sl->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active);
        Sl->tap_comm->communicate(ml->levels[i]->x);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_spmv_tests; test++)
        {
            // b <- A*x
            Sl->tap_comm->communicate(ml->levels[i]->x);
        }
        tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);

        // Vector communication with P
        MPI_Barrier(MPI_COMM_WORLD);
        ml->levels[i+1]->x.set_const_value(0.0);
        if (rank == 0) printf("P Vector TAP Communication\n");
        Pl->print_tap_mult(proc_distances);
        active = 1;
        if (Pl->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active);
        Pl->tap_comm->communicate(ml->levels[i+1]->x);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_spmv_tests; test++)
        {
            // b <- A*x
            Pl->tap_comm->communicate(ml->levels[i+1]->x);
        }
        tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);




        delete Sl;
    }

    // Delete raptor hierarchy
    delete ml;
    delete A;
    MPI_Finalize();

    return 0;
}


