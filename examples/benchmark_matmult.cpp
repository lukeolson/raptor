// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "clear_cache.hpp"

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "core/types.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_matrix_IO.hpp"

#ifdef USING_MFEM
#include "gallery/external/mfem_wrapper.hpp"
#endif

// using namespace raptor;
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char* filename = argv[1];

    if (rank == 0) printf("Reading in A\n");
    ParCSRMatrix* A = readParMatrix(filename);
    if (!A->comm)
        A->comm = new ParComm(A->partition, A->off_proc_column_map);
    if (!A->tap_comm)
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map);

    ParCSRMatrix* T = (ParCSRMatrix*) (A->transpose());
    if (!T->comm)
        T->comm = new ParComm(T->partition, T->off_proc_column_map);
    if (!T->tap_comm)
        T->tap_comm = new TAPComm(T->partition, T->off_proc_column_map);

    ParCSRMatrix* B = new ParCSRMatrix(A);
    if (!B->comm)
        B->comm = new ParComm(B->partition, B->off_proc_column_map);
    if (!B->tap_comm)
        B->tap_comm = new TAPComm(B->partition, B->off_proc_column_map);



if (A->global_num_rows != T->global_num_cols) printf("Dif dims\n");
if (A->global_num_cols != T->global_num_rows) printf("Dif dims\n");
if (A->partition->first_local_row != T->partition->first_local_col) printf("Diff dims\n");
if (A->partition->first_local_col != T->partition->first_local_row) printf("Diff dims\n");
if (A->local_num_rows != T->on_proc_num_cols) printf("Different local dims\n");
if (A->on_proc_num_cols != T->local_num_rows) printf("Different local dims\n");

CSCMatrix* T_on = new CSCMatrix((CSRMatrix*)T->on_proc);
/*for (int i = 0; i < A->local_num_rows; i++)
{
    if (A->on_proc->idx1[i+1] != T_on->idx1[i+1]) printf("Diff indptr\n");
    for (int j = A->on_proc->idx1[i]; j < A->on_proc->idx1[i+1]; j++)
    {
        if (A->on_proc->idx2[j] != T_on->idx2[j]) printf("Different indices\n");
        if (fabs(A->on_proc->vals[j] - T_on->vals[j]) > zero_tol) printf("Different vals\n");
    } 
}*/

CSCMatrix* A_csc = new CSCMatrix((CSRMatrix*) A->off_proc);
for (int i = 0; i < A_csc->nnz; i++)
{
    int row = A_csc->idx2[i];
    A_csc->idx2[i] = A->local_row_map[row];
}    
CSRMatrix* recv_mat = A->comm->communicate_T(A_csc->idx1, A_csc->idx2, A_csc->vals, A->local_num_rows);
if (T->off_proc->nnz != recv_mat->nnz) printf("A->off_proc %d, recvmat %d\n", T->off_proc->nnz, recv_mat->nnz);
recv_mat->sort();
T->off_proc->sort();
for (int i = 0; i < A->local_num_rows; i++)
{
    if (T->off_proc->idx1[i+1] != recv_mat->idx1[i+1]) printf("Diff indptr\n");
    for (int j = T->off_proc->idx1[i]; j < T->off_proc->idx1[i+1]; j++)
    {
        if (T->off_proc_column_map[T->off_proc->idx2[j]] != recv_mat->idx2[j]) printf("Diff indices\n");
        if (fabs(T->off_proc->vals[j] - recv_mat->vals[j]) > zero_tol) printf("Diff values\n");
   
        if (T->off_proc_column_map[T->off_proc->idx2[j]] != recv_mat->idx2[j]
            ||  fabs(T->off_proc->vals[j] - recv_mat->vals[j]) > zero_tol) break;
    }
}


std::vector<int> send_procs(num_procs, 0);
std::vector<int> recv_procs(num_procs, 0);
for (int i = 0; i < T->comm->send_data->num_msgs; i++)
{
    int proc = T->comm->send_data->procs[i];
    send_procs[proc] = (T->comm->send_data->indptr[i+1] - T->comm->send_data->indptr[i]);
}
for (int i = 0; i < T->comm->recv_data->num_msgs; i++)
{
    int proc = T->comm->recv_data->procs[i];
    recv_procs[proc] = (T->comm->recv_data->indptr[i+1] - T->comm->recv_data->indptr[i]);
}

    long nnz = A->local_nnz;
    long sum_nnz;
    MPI_Reduce(&nnz, &sum_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("A %d x %d with %ld nnz\n", A->global_num_rows, A->global_num_cols,
            sum_nnz);

    double t0, tfinal;
    int num_tests = 1;
    ParCSRMatrix* C;

MPI_Barrier(MPI_COMM_WORLD);
if (rank == 0) printf("communicating\n");
CSRMatrix* recv_tmp = T->comm->communicate(A);
printf("RecvTmp %d, %d, %d\n", recv_tmp->n_rows, recv_tmp->n_cols, recv_tmp->nnz);
delete recv_tmp;


    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Warming up...\n");
    C = T->mult(A);
    delete C;
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Multiplying...\n");
    tfinal = 0;
    //for (int i = 0; i < num_tests; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        C = T->mult(A);
        tfinal += (MPI_Wtime() - t0);
        delete C;
    }
    tfinal /= num_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Row Wise SpGEMM: %e\n", t0);


    if (rank == 0) printf("Warming up...\n");
    C = T->tap_mult(A);
    delete C;

    if (rank == 0) printf("Multiplying...\n");
    tfinal = 0;
    for (int i = 0; i < num_tests; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        C = T->tap_mult(A);
        tfinal += (MPI_Wtime() - t0);
        delete C;
    }
    tfinal /= num_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Row Wise NAP SpGEMM: %e\n", t0);

    if (rank == 0) printf("Warming up...\n");
    C = A->mult_T(A);
    delete C;

    if (rank == 0) printf("Multiplying...\n");
    tfinal = 0;
    for (int i = 0; i < num_tests; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        C = A->mult_T(A);
        tfinal += (MPI_Wtime() - t0);
        delete C;
    }
    tfinal /= num_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Outer Product SpGEMM: %e\n", t0);

    if (rank == 0) printf("Warming up...\n");
    C = A->tap_mult_T(A);
    delete C;

    if (rank == 0) printf("Multiplying...\n");
    tfinal = 0;
    for (int i = 0; i < num_tests; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        C = A->tap_mult_T(A);
        tfinal += (MPI_Wtime() - t0);
        delete C;
    }
    tfinal /= num_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Outer Product NAP SpGEMM: %e\n", t0);



    delete A;
    delete B;
    delete T;

    MPI_Finalize();

    return 0;
}


