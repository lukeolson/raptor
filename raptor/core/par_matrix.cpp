// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_matrix.hpp"

using namespace raptor;

/**************************************************************
*****   ParMatrix Add Value
**************************************************************
***** Adds a value to the local portion of the parallel matrix,
***** determining whether it should be added to diagonal or 
***** off-diagonal block. 
*****
***** Parameters
***** -------------
***** row : index_t
*****    Local row of value
***** global_col : index_t 
*****    Global column of value
***** value : data_t
*****    Value to be added to parallel matrix
**************************************************************/    
void ParMatrix::add_value(
        int row, 
        index_t global_col, 
        data_t value)
{
    if (global_col >= partition->first_local_col 
            && global_col <= partition->last_local_col)
    {
        on_proc->add_value(row, global_col - partition->first_local_col, value);
    }
    else 
    {
        off_proc->add_value(row, global_col, value);
    }
}

/**************************************************************
*****   ParMatrix Add Global Value
**************************************************************
***** Adds a value to the local portion of the parallel matrix,
***** determining whether it should be added to diagonal or 
***** off-diagonal block. 
*****
***** Parameters
***** -------------
***** global_row : index_t
*****    Global row of value
***** global_col : index_t 
*****    Global column of value
***** value : data_t
*****    Value to be added to parallel matrix
**************************************************************/ 
void ParMatrix::add_global_value(
        index_t global_row, 
        index_t global_col, 
        data_t value)
{
    add_value(global_row - partition->first_local_row, global_col, value);
}

/**************************************************************
*****   ParMatrix Finalize
**************************************************************
***** Finalizes the diagonal and off-diagonal matrices.  Sorts
***** the local_to_global indices, and creates the parallel
***** communicator
*****
***** Parameters
***** -------------
***** create_comm : bool (optional)
*****    Boolean for whether parallel communicator should be 
*****    created (default is true)
**************************************************************/
void ParMatrix::condense_off_proc()
{
    if (off_proc->nnz == 0)
    {
        return;
    }

    int prev_col = -1;

    std::map<int, int> orig_to_new;

    std::copy(off_proc->idx2.begin(), off_proc->idx2.end(),
            std::back_inserter(off_proc_column_map));
    std::sort(off_proc_column_map.begin(), off_proc_column_map.end());

    off_proc_num_cols = 0;
    for (aligned_vector<int>::iterator it = off_proc_column_map.begin(); 
            it != off_proc_column_map.end(); ++it)
    {
        if (*it != prev_col)
        {
            orig_to_new[*it] = off_proc_num_cols;
            off_proc_column_map[off_proc_num_cols++] = *it;
            prev_col = *it;
        }
    }
    off_proc_column_map.resize(off_proc_num_cols);

    for (aligned_vector<int>::iterator it = off_proc->idx2.begin();
            it != off_proc->idx2.end(); ++it)
    {
        *it = orig_to_new[*it];
    }
}

// Expands the off_proc_column_map for BSR matrices to hold the
// global columns in off process with non-zeros, not just the
// coarse block columns
void ParMatrix::expand_off_proc(int b_cols)
{
    int start, end;
    aligned_vector<int> new_map;

    int rank, num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for(int i=0; i<off_proc_column_map.size(); i++)
    {
    start = off_proc_column_map[i] * b_cols;
    if (start >= partition->first_local_col && rank!= 0) start += partition->local_num_cols;
    end = start + b_cols;
        for(int j=start; j<end; j++)
    {
            new_map.push_back(j);
    }
    }

    off_proc_column_map.clear();
    std::copy(new_map.begin(), new_map.end(), std::back_inserter(off_proc_column_map));
}

void ParMatrix::finalize(bool create_comm, int b_cols)
{
    on_proc->sort();
    off_proc->sort();

    int rank, num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Assume nonzeros in each on_proc column
    if (on_proc_num_cols > on_proc_column_map.size())
    {
        on_proc_column_map.resize(on_proc_num_cols);
        for (int i = 0; i < on_proc_num_cols; i++)
        {
            on_proc_column_map[i] = i + partition->first_local_col;
        }
    }

    if (local_num_rows > local_row_map.size())
    {
        local_row_map.resize(local_num_rows);
        for (int i = 0; i < local_num_rows; i++)
        {
            local_row_map[i] = i + partition->first_local_row;
        }
    }

    // Condense columns in off_proc, storing global
    // columns as 0-num_cols, and store mapping
    if (off_proc->nnz)
    {
        condense_off_proc();
    }
    else
    {
        off_proc_num_cols = 0;
    }
    off_proc->resize(local_num_rows, off_proc_num_cols);
    local_nnz = on_proc->nnz + off_proc->nnz;

    // If BSR matrix - correct the off_proc_column_map
    // to include all global columns within block
    if (b_cols)
    {
        expand_off_proc(b_cols);
        if (rank == 0)
    {
        for (int i = 0; i < off_proc_column_map.size(); i++)
        {
                off_proc_column_map[i] += on_proc_num_cols;
            }
        }
    }

    if (create_comm){
        comm = new ParComm(partition, off_proc_column_map);
    }
    else
        comm = new ParComm(partition);
}

int* ParMatrix::map_partition_to_local()
{
    int* on_proc_partition_to_col = new int[partition->local_num_cols+1];
    for (int i = 0; i < partition->local_num_cols+1; i++) on_proc_partition_to_col[i] = -1;
    for (int i = 0; i < on_proc_num_cols; i++)
    {
        on_proc_partition_to_col[on_proc_column_map[i] - partition->first_local_col] = i;
    }

    return on_proc_partition_to_col;
}

void ParMatrix::copy(ParCOOMatrix* A)
{
    partition = A->partition;
    partition->num_shared++;

    local_nnz = A->local_nnz;
    local_num_rows = A->local_num_rows;
    global_num_rows = A->global_num_rows;
    global_num_cols = A->global_num_cols;

    std::copy(A->off_proc_column_map.begin(), A->off_proc_column_map.end(),
            std::back_inserter(off_proc_column_map));
    std::copy(A->on_proc_column_map.begin(), A->on_proc_column_map.end(),
            std::back_inserter(on_proc_column_map));
    std::copy(A->local_row_map.begin(), A->local_row_map.end(),
            std::back_inserter(local_row_map));

    off_proc_num_cols = off_proc_column_map.size();
    on_proc_num_cols = on_proc_column_map.size();

    if (A->comm)
    {
        comm = new ParComm((ParComm*) A->comm);
    }
    else
    {   
        comm = NULL;
    }
    
    if (A->tap_comm)
    {
        tap_comm = new TAPComm((TAPComm*) A->tap_comm);
    }
    else
    {
        tap_comm = NULL;
    }
}

void ParMatrix::copy(ParCSRMatrix* A)
{
    partition = A->partition;
    partition->num_shared++;

    local_nnz = A->local_nnz;
    local_num_rows = A->local_num_rows;
    global_num_rows = A->global_num_rows;
    global_num_cols = A->global_num_cols;

    std::copy(A->off_proc_column_map.begin(), A->off_proc_column_map.end(),
            std::back_inserter(off_proc_column_map));
    std::copy(A->on_proc_column_map.begin(), A->on_proc_column_map.end(),
            std::back_inserter(on_proc_column_map));
    std::copy(A->local_row_map.begin(), A->local_row_map.end(),
            std::back_inserter(local_row_map));

    off_proc_num_cols = off_proc_column_map.size();
    on_proc_num_cols = on_proc_column_map.size();

    if (A->comm)
    {
        comm = new ParComm((ParComm*) A->comm);
    }
    else
    {   
        comm = NULL;
    }
    
    if (A->tap_comm)
    {
        tap_comm = new TAPComm((TAPComm*) A->tap_comm);
    }
    else
    {
        tap_comm = NULL;
    }
}

void ParMatrix::copy(ParCSCMatrix* A)
{
    partition = A->partition;
    partition->num_shared++;

    local_nnz = A->local_nnz;
    local_num_rows = A->local_num_rows;
    global_num_rows = A->global_num_rows;
    global_num_cols = A->global_num_cols;

    std::copy(A->off_proc_column_map.begin(), A->off_proc_column_map.end(),
            std::back_inserter(off_proc_column_map));
    std::copy(A->on_proc_column_map.begin(), A->on_proc_column_map.end(),
            std::back_inserter(on_proc_column_map));
    std::copy(A->local_row_map.begin(), A->local_row_map.end(),
            std::back_inserter(local_row_map));

    off_proc_num_cols = off_proc_column_map.size();
    on_proc_num_cols = on_proc_column_map.size();
    
    if (A->comm)
    {
        comm = new ParComm((ParComm*) A->comm);
    }
    else
    {   
        comm = NULL;
    }
    
    if (A->tap_comm)
    {
        tap_comm = new TAPComm((TAPComm*) A->tap_comm);
    }
    else
    {
        tap_comm = NULL;
    }
}

void ParMatrix::copy(ParBSRMatrix* A)
{
    partition = A->partition;
    partition->num_shared++;

    local_nnz = A->local_nnz;
    local_num_rows = A->local_num_rows;
    global_num_rows = A->global_num_rows;
    global_num_cols = A->global_num_cols;

    std::copy(A->off_proc_column_map.begin(), A->off_proc_column_map.end(),
            std::back_inserter(off_proc_column_map));
    std::copy(A->on_proc_column_map.begin(), A->on_proc_column_map.end(),
            std::back_inserter(on_proc_column_map));
    std::copy(A->local_row_map.begin(), A->local_row_map.end(),
            std::back_inserter(local_row_map));

    off_proc_num_cols = off_proc_column_map.size();
    on_proc_num_cols = on_proc_column_map.size();

    if (A->comm)
    {
        comm = new ParComm((ParComm*) A->comm);
    }
    else
    {   
        comm = NULL;
    }
    
    if (A->tap_comm)
    {
        tap_comm = new TAPComm((TAPComm*) A->tap_comm);
    }
    else
    {
        tap_comm = NULL;
    }
}

void ParCOOMatrix::copy(ParCSRMatrix* A)
{
    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new COOMatrix((CSRMatrix*) A->on_proc);
    off_proc = new COOMatrix((CSRMatrix*) A->off_proc);

    ParMatrix::copy(A);
}

void ParCOOMatrix::copy(ParCSCMatrix* A)
{
    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new COOMatrix((CSCMatrix*) A->on_proc);
    off_proc = new COOMatrix((CSCMatrix*) A->off_proc);

    ParMatrix::copy(A);
}

void ParCOOMatrix::copy(ParCOOMatrix* A)
{
    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new COOMatrix((COOMatrix*) A->on_proc);
    off_proc = new COOMatrix((COOMatrix*) A->off_proc);

    ParMatrix::copy(A);
}

void ParCOOMatrix::copy(ParBSRMatrix* A)
{
    if (on_proc)
    {
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new COOMatrix((COOMatrix*) A->on_proc);
    off_proc = new COOMatrix((COOMatrix*) A->off_proc);

    ParMatrix::copy(A);
}

void ParCSRMatrix::copy(ParCSRMatrix* A)
{
    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }
    on_proc = new CSRMatrix((CSRMatrix*) A->on_proc);
    off_proc = new CSRMatrix((CSRMatrix*) A->off_proc);

    ParMatrix::copy(A);
}

void ParCSRMatrix::copy(ParCSCMatrix* A)
{
    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }
    on_proc = new CSRMatrix((CSCMatrix*) A->on_proc);
    off_proc = new CSRMatrix((CSCMatrix*) A->off_proc);

    ParMatrix::copy(A);
}

void ParCSRMatrix::copy(ParCOOMatrix* A)
{
    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new CSRMatrix((COOMatrix*) A->on_proc);
    off_proc = new CSRMatrix((COOMatrix*) A->off_proc);

    ParMatrix::copy(A);
}

void ParCSRMatrix::copy(ParBSRMatrix* A)
{
    printf("Currently not implemented\n");
}

void ParCSCMatrix::copy(ParCSRMatrix* A)
{
    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new CSCMatrix((CSRMatrix*) A->on_proc);
    off_proc = new CSCMatrix((CSRMatrix*) A->off_proc);

    ParMatrix::copy(A);
}

void ParCSCMatrix::copy(ParCSCMatrix* A)
{
    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new CSCMatrix((CSCMatrix*) A->on_proc);
    off_proc = new CSCMatrix((CSCMatrix*) A->off_proc);

    ParMatrix::copy(A);
}

void ParCSCMatrix::copy(ParCOOMatrix* A)
{
    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new CSCMatrix((COOMatrix*) A->on_proc);
    off_proc = new CSCMatrix((COOMatrix*) A->off_proc);

    ParMatrix::copy(A);
}

void ParCSCMatrix::copy(ParBSRMatrix* A)
{
    printf("Currently not implemented\n");
}

void ParBSRMatrix::copy(ParCSRMatrix* A)
{
    printf("Currently not implemented.\n");
}

void ParBSRMatrix::copy(ParCSCMatrix* A)
{
    printf("Currently not implemented.\n");
}

void ParBSRMatrix::copy(ParCOOMatrix* A)
{
    printf("Currently not implemented.\n");
}

void ParBSRMatrix::copy(ParBSRMatrix* A)
{
    printf("Currently not implemented\n");
}


void ParCOOMatrix::add_block(int global_row_coarse, int global_col_coarse, aligned_vector<double>& data){
    printf("currently not implemented.\n");
}

void ParCSRMatrix::add_block(int global_row_coarse, int global_col_coarse, aligned_vector<double>& data){
    printf("currently not implemented.\n");
}

void ParCSCMatrix::add_block(int global_row_coarse, int global_col_coarse, aligned_vector<double>& data){
    printf("currently not implemented.\n");
}

/***********************************************************
***** ParBSRMatrix::add_block()
************************************************************
***** Input:
*****    global_row_coarse:
*****        row index of block in coarse global block matrix
*****    global_col_coarse:
*****        colum index of block in coarse global block matrix
*****    data:
*****        vector of values for non-zero block to be added
***********************************************************/
void ParBSRMatrix::add_block(int global_row_coarse, int global_col_coarse, aligned_vector<double>& data){

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Takes global row and global column of block in coarse matrix
    if(num_procs <= 1)
    {
        on_proc->add_block(global_row_coarse, global_col_coarse, data);
    local_nnz += b_size;
    }
    else
    {
        // Global indices for block
        int first_row = global_row_coarse * b_rows; // first global row
        int last_row = first_row + b_rows - 1; // last global row
        int first_col = global_col_coarse * b_cols; // first global col
        int last_col = first_col + b_cols - 1; // last global col 

        // local block row index in coarse matrix
        int local_block_row = global_row_coarse % (partition->local_num_rows / b_rows);
        int local_block_col; // local block col index in coarse matrix

        // Check if block belongs to this process - then add to on_proc or off_proc
        if (first_row >= partition->first_local_row &&
            last_row <= partition->last_local_row)
        {
            if (first_col >= partition->first_local_col &&
                last_col <= partition->last_local_col)
            {
            local_block_col = global_col_coarse % (partition->local_num_cols / b_cols);
                on_proc->add_block(local_block_row, local_block_col, data);
        }
        else
        {
                // Check to see if block is before on_proc columns or after to 
            // determine whether local_block_col changes or stays the same
                if (last_col < partition->last_local_col) local_block_col = global_col_coarse;
                else local_block_col = global_col_coarse - (partition->local_num_cols / b_cols);
                off_proc->add_block(local_block_row, local_block_col, data);
        }

        // Update local nnz
        local_nnz += b_size;
        }
    }
}


ParMatrix* ParCOOMatrix::transpose()
{
    // NOT IMPLEMENTED
    return NULL;
}

ParMatrix* ParCSRMatrix::transpose()
{
    int start, end;
    int proc;
    int col, col_start, col_end;
    int ctr, size;
    int col_count, count;
    int col_size;
    int idx, row;
    MPI_Status recv_status;

    if (!comm) comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);

    Partition* part_T;
    Matrix* on_proc_T;
    Matrix* off_proc_T;
    CSCMatrix* send_mat;
    ParCSRMatrix* T = NULL;

    aligned_vector<PairData> send_buffer;
    aligned_vector<PairData> recv_buffer;
    aligned_vector<int> send_ptr(comm->recv_data->num_msgs+1);

    // Transpose partition
    part_T = partition->transpose();

    // Transpose local (on_proc) matrix
    on_proc_T = on_proc->transpose();

    // Allocate vectors for sending off_proc matrix
    send_mat = new CSCMatrix((CSRMatrix*) off_proc);

    for (int i = 0; i < send_mat->nnz; i++)
    {
        int row = send_mat->idx2[i];
        send_mat->idx2[i] = local_row_map[row];
    }
    off_proc_T = comm->communicate_T(send_mat->idx1, send_mat->idx2, send_mat->vals, local_num_rows);

    T = new ParCSRMatrix(part_T, on_proc_T, off_proc_T);

    delete send_mat;

    return T;
}

ParMatrix* ParCSCMatrix::transpose()
{
    // NOT IMPLEMENTED
    return NULL;
}

ParMatrix* ParBSRMatrix::transpose()
{
    // NOT IMPLEMENTED
    return NULL;
}
