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
#include "tests/hypre_compare.hpp"

#ifdef USING_MFEM
#include "gallery/external/mfem_wrapper.hpp"
#endif

//using namespace raptor;
int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim;
    int n = 5;
    int system = 0;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;

    double strong_threshold = 0.25;
    int cache_len = 10000;
    int num_tests = 2;
    std::vector<double> cache_array(cache_len);

    if (system < 2)
    {
        double* stencil = NULL;
        std::vector<int> grid;
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
            double theta = M_PI/8.0;
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
        char* mesh_file = argv[2];
        int num_elements = 2;
        int order = 3;
        if (argc > 3)
        {
            num_elements = atoi(argv[3]);
            if (argc > 4)
            {
                order = atoi(argv[4]);
            }
        }
        A = mfem_linear_elasticity(x, b, mesh_file, num_elements, order);
    }
#endif
    else if (system == 3)
    {
        char* file = "../../examples/LFAT5.mtx";
        int sym = 1;
        if (argc > 2)
        {
            file = argv[2];
            if (argc > 3)
            {
                sym = atoi(argv[3]);
            }
        }
        A = readParMatrix(file, MPI_COMM_WORLD, 1, sym);
    }

    if (system != 2)
    {
        x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
        b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
        x.set_const_value(1.0);
        A->mult(x, b);
    }

    ParMultilevel* ml;

    x.set_const_value(0.0);

    // Setup Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);    
    ml = new ParMultilevel(A, strong_threshold);

    long lcl_nnz;
    long nnz;
    if (rank == 0) printf("Level\tNumRows\tNNZ\n");
    for (int i = 0; i < ml->num_levels; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        lcl_nnz = Al->local_nnz;
        MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%d\t%d\t%ld\n", i, Al->global_num_rows, nnz);
    }   

    for (int i = 0; i < ml->num_levels; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParVector& xl = ml->levels[i]->x;
        ParVector& bl = ml->levels[i]->b;
        xl.set_const_value(1.0);

        // Add everything to cache... only measuring comm time
        Al->mult(xl, bl);

        t0 = MPI_Wtime();
        for (int i = 0; i < num_tests; i++)
        {
            Al->comm->communicate(xl.local.data());
        }
        tfinal = (MPI_Wtime() - t0) / num_tests;

        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Communication Time: %e\n", t0);

        int on_socket_n[3] = {0};
        int on_socket_s[3] = {0};
        int off_socket_n[3] = {0};
        int off_socket_s[3] = {0};
        int off_node_n[3] = {0};
        int off_node_s[3] = {0};

        int* comm_n;
        int* comm_s;
        rank_node = Al->partition->topology->get_node(rank);
        rank_socket = Al->partition->topology->get_local_rank(rank) / 8;
        for (int i = 0; i < Al->comm->send_data->num_msgs; i++)
        {
            proc = Al->comm->send_data->procs[i];
            proc_node = Al->partition->topology->get_node(proc);
            proc_socket = Al->partition->topology->get_local_rank(rank) / 8;
            start = Al->comm->send_data->indptr[i];
            end = Al->comm->send_data->indptr[i+1];
            size  = (end - start) * 8;
            if (rank_node == proc_node)
            {
                if (rank_socket == proc_socket)
                {
                    comm_n = on_socket_n;
                    comm_s = on_socket_s;
                }
                else
                {
                    comm_n = off_socket_n;
                    comm_s = off_socket_s;
                }
            }
            else
            {
                comm_n = off_node_n;
                comm_s = off_node_s;
            }
            if (size < short_cutoff)
            {
                comm_n[0]++;
                comm_s[0] += size;
            }
            else if (size < eager_cutoff)
            {
                comm_n[1]++;
                comm_s[1] += size;
            }
            else
            {
                comm_n[2]++;
                comm_s[2] += size;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &on_socket_s, 3, MPI_INT, MPI_SUM, Al->partition->topology->local_comm);
        MPI_Allreduce(MPI_IN_PLACE, &off_socket_s, 3, MPI_INT, MPI_SUM, Al->partition->topology->local_comm);
        MPI_Allreduce(MPI_IN_PLACE, &off_node_s, 3, MPI_INT, MPI_SUM, Al->partition->topology->local_comm);

    }

    delete ml;
    delete A;

    MPI_Finalize();

    return 0;
}

