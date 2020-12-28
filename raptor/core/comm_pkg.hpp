// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PARCOMM_HPP
#define RAPTOR_CORE_PARCOMM_HPP

#include <mpi.h>
#include "comm_data.hpp"
#include "matrix.hpp"
#include "partition.hpp"
#include "par_vector.hpp"

#define STANDARD_PPN 4
#define STANDARD_PROC_LAYOUT 1

/**************************************************************
 *****   CommPkg Class:
 **************************************************************
 ***** This class constructs a parallel communicator, containing
 ***** which messages must be sent/recieved for matrix operations
 *****
 ***** Methods
 ***** -------
 ***** communicate(data_t* values)
 *****    Communicates values to processes, based on underlying
 *****    communication package
 ***** form_col_to_proc(...)
 *****    Maps each column in off_proc_column_map to process 
 *****    on which corresponding values are stored
 **************************************************************/
namespace raptor
{
    class ParCSRMatrix;
    class ParBSRMatrix;

    class CommPkg
    {
      public:
        CommPkg(Partition* partition)
        {
            topology = partition->topology;
            topology->num_shared++;
            num_shared = 0;
        }
        
        CommPkg(Topology* _topology)
        {
            topology = _topology;
            topology->num_shared++;
            num_shared = 0;
        }

        virtual ~CommPkg()
        {
            if (topology)
            {
                if (topology->num_shared)
                {
                    topology->num_shared--;
                }
                else
                {
                    delete topology;
                }
            }
        }

        void delete_comm()
        {
            if (num_shared == 0)
                delete this;
            else num_shared--;
        }

        // Matrix Communication
        // TODO -- Block transpose communication
        //      -- Should b_rows / b_cols be switched?
        virtual CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true) = 0;
        virtual CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true) = 0;
        virtual void init_mat_comm(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true) = 0;
        virtual void init_mat_comm(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true) = 0;
        virtual CSRMatrix* complete_mat_comm(const int b_rows = 1, const int b_cols = 1, 
                const bool has_vals = true) = 0;

        virtual CSRMatrix* communicate_T(const aligned_vector<int>& rowptr,
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values, 
                const int n_result_rows, const int b_rows = 1, const int b_cols = 1,
                const bool has_vals = true) = 0;
        virtual CSRMatrix* communicate_T(const aligned_vector<int>& rowptr,
                const aligned_vector<int>& col_indices, const aligned_vector<double*>& values, 
                const int n_result_rows, const int b_rows = 1, const int b_cols = 1,
                const bool has_vals = true) = 0;
        virtual void init_mat_comm_T(aligned_vector<char>& send_buffer, 
                const aligned_vector<int>& rowptr, const aligned_vector<int>& col_indices, 
                const aligned_vector<double>& values, const int b_rows = 1, 
                const int b_cols = 1, const bool has_vals = true) = 0;
        virtual void init_mat_comm_T(aligned_vector<char>& send_buffer,
                const aligned_vector<int>& rowptr, const aligned_vector<int>& col_indices, 
                const aligned_vector<double*>& values, const int b_rows = 1, 
                const int b_cols = 1, const bool has_vals = true) = 0;
        virtual CSRMatrix* complete_mat_comm_T(const int n_result_rows, 
                const int b_rows = 1, const int b_cols = 1,
                const bool has_vals = true) = 0;

        aligned_vector<double>& get_vals(CSRMatrix* A)
        {
            return A->vals;
        }
        aligned_vector<double*> get_vals(BSRMatrix* A)
        {
            return A->block_vals;
        }

        CSRMatrix* communicate_sparsity(ParCSRMatrix* A)
        {
            return communicate(A, false);
        }

        CSRMatrix* communicate(ParCSRMatrix* A, const bool has_vals = true);
        CSRMatrix* communicate(ParBSRMatrix* A, const bool has_vals = true);
        void init_par_mat_comm(ParCSRMatrix* A, aligned_vector<char>& send_buffer,
                const bool has_vals = true);
        void init_par_mat_comm(ParBSRMatrix* A, aligned_vector<char>& send_buffer,
                const bool has_vals = true);

        CSRMatrix* communicate(CSRMatrix* A, const int has_vals = true)
        {
            return communicate(A->idx1, A->idx2, get_vals(A), A->b_rows, A->b_cols, has_vals);
        }
        CSRMatrix* communicate_T(CSRMatrix* A, const int has_vals = true)
        {
            return communicate_T(A->idx1, A->idx2, get_vals(A), A->n_rows, A->b_rows, 
                    A->b_cols, has_vals);
        }

        // Vector Communication
        aligned_vector<double>& communicate(ParVector& v, const int block_size = 1);
        void init_comm(ParVector& v, const int block_size = 1, const int vblock_size = 1);

        // Standard Communication
        template<typename T>
        aligned_vector<T>& communicate(const aligned_vector<T>& values, const int block_size = 1,
                                       const int vblock_size = 1, const int vblock_offset = 0)
        {
            //return communicate(values.data(), block_size, vblock_size, vblock_offset);
            return communicate(values.data(), block_size);
        }
        template<typename T>
        void init_comm(const aligned_vector<T>& values, const int block_size = 1, const int vblock_size = 1)
        {
            init_comm(values.data(), block_size, vblock_size);
        }
        template<typename T> void init_comm(const T* values, const int block_size = 1,
                const int vblock_size = 1);
        template<typename T> aligned_vector<T>& complete_comm(const int block_size = 1,
                const int vblock_size = 1);
        template<typename T> aligned_vector<T>& communicate(const T* values, const int block_size = 1,
                const int vblock_size = 1, const int vblock_offset = 0);
        virtual void init_double_comm(const double* values, const int block_size,
                const int vblock_size = 1, const int vblock_offset = 0) = 0;
        virtual void init_int_comm(const int* values, const int block_size,
                const int vblock_size = 1) = 0;
        virtual aligned_vector<double>& complete_double_comm(const int block_size,
                const int vblock_size = 1) = 0;
        virtual aligned_vector<int>& complete_int_comm(const int block_size,
                const int vblock_size = 1) = 0;

        // Transpose Communication
        template<typename T, typename U>
        void communicate_T(const aligned_vector<T>& values, aligned_vector<U>& result,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<U(U, T)> result_func = &sum_func<T, U>,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        { 
            communicate_T(values.data(), result, block_size, vblock_size, vblock_offset, result_func, 
                    init_result_func, init_result_func_val);
        }
        template<typename T>
        void communicate_T(const aligned_vector<T>& values,
                const int block_size = 1, 
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            communicate_T(values.data(), block_size, vblock_size, vblock_offset, init_result_func,
                    init_result_func_val);
        }
        template<typename T>
        void init_comm_T(const aligned_vector<T>& values,
                const int block_size = 1, 
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            init_comm_T(values.data(), block_size, vblock_size, vblock_offset, init_result_func, init_result_func_val);
        }
        template<typename T> void init_comm_T(const T* values,
                const int block_size = 1, 
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<T(T, T)> init_result_func = &sum_func<T, T>, 
                T init_result_func_val = 0);
        template<typename T, typename U> void complete_comm_T(aligned_vector<U>& result,
                const int block_size = 1,
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<U(U, T)> result_func = &sum_func<T, U>,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>, 
                T init_result_func_val = 0);
        template<typename T> void complete_comm_T(
                const int block_size = 1,
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0);
        template<typename T, typename U> void communicate_T(const T* values, 
                aligned_vector<U>& result, const int block_size = 1, 
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<U(U, T)> result_func = &sum_func<T, U>,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>, 
                T init_result_func_val = 0);
        template<typename T> void communicate_T(const T* values,
                const int block_size = 1,
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0);
        virtual void init_double_comm_T(const double* values,
                const int block_size,
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<double(double, double)> init_result_func = 
                    &sum_func<double, double>, 
                    double init_result_func_val = 0) = 0;
        virtual void init_int_comm_T(const int* values,
                const int block_size,
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<int(int, int)> init_result_func = &sum_func<int, int>,
                int init_result_func_val = 0) = 0;
        virtual void complete_double_comm_T(aligned_vector<double>& result,
                const int block_size,
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<double(double, double)> result_func = &sum_func<double, double>,
                std::function<double(double, double)> init_result_func = 
                    &sum_func<double, double>, double init_result_func_val = 0) = 0;
        virtual void complete_double_comm_T(aligned_vector<int>& result,
                const int block_size,
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<int(int, double)> result_func = &sum_func<double, int>,
                std::function<double(double, double)> init_result_func = 
                    &sum_func<double, double>, double init_result_func_val = 0) = 0;
        virtual void complete_int_comm_T(aligned_vector<int>& result,
                const int block_size,
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<int(int, int)> result_func = &sum_func<int, int>,
                std::function<int(int, int)> init_result_func = &sum_func<int, int>,
                int init_result_func_val = 0) = 0;
        virtual void complete_int_comm_T(aligned_vector<double>& result,
                const int block_size,
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<double(double, int)> result_func = &sum_func<int, double>,
                std::function<int(int, int)> init_result_func = &sum_func<int, int>,
                int init_result_func_val = 0) = 0;
        virtual void complete_double_comm_T(const int block_size,
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<double(double, double)> init_result_func = 
                    &sum_func<double, double>, double init_result_func_val = 0) = 0;
        virtual void complete_int_comm_T(const int block_size,
                const int vblock_size = 1, 
                const int vblock_offset = 0, 
                std::function<int(int, int)> init_result_func = &sum_func<int, int>,
                int init_result_func_val = 0) = 0;

        // Helper methods
        template <typename T> aligned_vector<T>& get_buffer();
        virtual aligned_vector<double>& get_double_buffer() = 0;
        virtual aligned_vector<int>& get_int_buffer() = 0;

        // Class Variables
        Topology* topology;
        aligned_vector<double> buffer;
        aligned_vector<int> int_buffer;
        int num_shared;
    };


    /**************************************************************
    *****   ParComm Class
    **************************************************************
    ***** This class constructs a standard parallel communicator: 
    ***** which messages must be sent/recieved for matrix operations
    *****
    ***** Attributes
    ***** -------------
    ***** num_sends : index_t
    *****    Number of messages this process must send during 
    *****    matrix operations
    ***** num_recvs : index_t
    *****    Number of messages this process will recv during
    *****    matrix operations
    ***** size_sends : index_t 
    *****    Total number of elements this process sends in all
    *****    messages
    ***** size_recvs : index_t 
    *****    Total number of elements this process recvs from
    *****    all messages
    ***** send_procs : aligned_vector<int>
    *****    Distant processes messages are to be sent to
    ***** send_row_starts : aligned_vector<int>
    *****    Pointer to first position in send_row_indices
    *****    that a given process will send.
    ***** send_row_indices : aligned_vector<int> 
    *****    The indices of values that must be sent to each
    *****    process in send_procs
    ***** recv_procs : aligned_vector<int>
    *****    Distant processes messages are to be recvd from
    ***** recv_col_starts : aligned_vector<int>
    *****    Pointer to first column recvd from each process
    *****    in recv_procs
    ***** col_to_proc : aligned_vector<int>
    *****    Maps each local column in the off-diagonal block
    *****    to the process that holds corresponding data
    **************************************************************/
    class ParComm : public CommPkg
    {
      public:
        /**************************************************************
        *****   ParComm Class Constructor
        **************************************************************
        ***** Initializes an empty ParComm, setting send and recv
        ***** sizes to 0
        *****
        ***** Parameters
        ***** -------------
        ***** _key : int (optional)
        *****    Tag to be used in RAPtor_MPI Communication (default 0)
        **************************************************************/
        ParComm(Partition* partition, int _key = 0, 
                RAPtor_MPI_Comm _comm = RAPtor_MPI_COMM_WORLD,
                CommData* r_data = NULL) : CommPkg(partition)
        {
            mpi_comm = _comm;
            key = _key;
            send_data = new NonContigData();
            if (r_data)
                recv_data = r_data;
            else
                recv_data = new ContigData();
        }

        ParComm(Topology* topology, int _key = 0, 
                RAPtor_MPI_Comm _comm = RAPtor_MPI_COMM_WORLD,
                CommData* r_data = NULL) : CommPkg(topology)
        {
            mpi_comm = _comm;
            key = _key;
            send_data = new NonContigData();
            if (r_data)
                recv_data = r_data;
            else
                recv_data = new ContigData();
        }

        /**************************************************************
        *****   ParComm Class Constructor
        **************************************************************
        ***** Initializes a ParComm object based on the off_proc Matrix
        *****
        ***** Parameters
        ***** -------------
        ***** off_proc_column_map : aligned_vector<int>&
        *****    Maps local off_proc columns indices to global
        ***** _key : int (optional)
        *****    Tag to be used in RAPtor_MPI Communication (default 9999)
        **************************************************************/
        ParComm(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                int _key = 9999,
                RAPtor_MPI_Comm comm = RAPtor_MPI_COMM_WORLD,
                CommData* r_data = NULL) : CommPkg(partition)
        {
            mpi_comm = comm;
            init_par_comm(partition, off_proc_column_map, _key, comm, r_data);
        }

        ParComm(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                const aligned_vector<int>& on_proc_column_map,
                int _key = 9999, 
                RAPtor_MPI_Comm comm = RAPtor_MPI_COMM_WORLD,
                CommData* r_data = NULL) : CommPkg(partition)
        {
            mpi_comm = comm;
            int idx;
            int ctr = 0;
            aligned_vector<int> part_col_to_new;

            init_par_comm(partition, off_proc_column_map, _key, comm, r_data);

            if (partition->local_num_cols)
            {
                part_col_to_new.resize(partition->local_num_cols, -1);
            }
            for (aligned_vector<int>::const_iterator it = on_proc_column_map.begin();
                    it != on_proc_column_map.end(); ++it)
            {
                part_col_to_new[*it - partition->first_local_col] = ctr++;
            }

            for (int i = 0; i < send_data->size_msgs; i++)
            {
                idx = send_data->indices[i];
                send_data->indices[i] = part_col_to_new[idx];
                assert(part_col_to_new[idx] >= 0);
            }

	    
        }

        void init_par_comm(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                int _key, RAPtor_MPI_Comm comm,
                CommData* r_data = NULL)
        {
            // Get RAPtor_MPI Information
            int rank, num_procs;
            RAPtor_MPI_Comm_rank(comm, &rank);
            RAPtor_MPI_Comm_size(comm, &num_procs);

            // Initialize class variables
            key = _key;

            send_data = new NonContigData();

            if (r_data)
                recv_data = r_data;
            else
                recv_data = new ContigData();

            // Declare communication variables
            int send_start, send_end;
            int proc, prev_proc;
            int count;
            int tag = 12345;  // TODO -- switch this to key?
            int off_proc_num_cols = off_proc_column_map.size();
            RAPtor_MPI_Status recv_status;

            aligned_vector<int> off_proc_col_to_proc(off_proc_num_cols);
            aligned_vector<int> tmp_send_buffer;

            partition->form_col_to_proc(off_proc_column_map, off_proc_col_to_proc);

            // Determine processes columns are received from,
            // and adds corresponding messages to recv data.
            // Assumes columns are partitioned across processes
            // in contiguous blocks, and are sorted
            if (off_proc_num_cols)
            {
                prev_proc = off_proc_col_to_proc[0];
                int prev_idx = 0;
                for (int i = 1; i < off_proc_num_cols; i++)
                {
                    proc = off_proc_col_to_proc[i];
                    if (proc != prev_proc)
                    {
                        recv_data->add_msg(prev_proc, i - prev_idx);
                        prev_proc = proc;
                        prev_idx = i;
                    }
                }
                recv_data->add_msg(prev_proc, off_proc_num_cols - prev_idx);
                recv_data->finalize();
            }

            // For each process I recv from, send the global column indices
            // for which I must recv corresponding rows 
            aligned_vector<int> recv_sizes(num_procs, 0);
            for (int i = 0; i < recv_data->num_msgs; i++)
                recv_sizes[recv_data->procs[i]] = 
                    recv_data->indptr[i+1] - recv_data->indptr[i];
            RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, recv_sizes.data(), num_procs, RAPtor_MPI_INT,
                    RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
            if (profile) vec_t -= RAPtor_MPI_Wtime();
            recv_data->send(off_proc_column_map.data(), tag, comm);
            send_data->probe(recv_sizes[rank], tag, comm);
            recv_data->waitall();
            if (profile) vec_t += RAPtor_MPI_Wtime();
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
            for (int i = 0; i < send_data->size_msgs; i++)
            {
                send_data->indices[i] -= partition->first_local_col;
            }
        }

        ParComm(ParComm* comm) : CommPkg(comm->topology)
        {
            mpi_comm = comm->mpi_comm;
            send_data = comm->send_data->copy();
            recv_data = comm->recv_data->copy();
            key = comm->key;
        }

        ParComm(ParComm* comm, const aligned_vector<int>& off_proc_col_to_new)
            : CommPkg(comm->topology)
        {
            mpi_comm = comm->mpi_comm;
            bool comm_proc;
            int proc, start, end;
            int idx, new_idx;

            if (comm == NULL)
            {
                key = 0;
                return;
            }
            key = comm->key;

            init_off_proc_new(comm, off_proc_col_to_new);
        }
        
        ParComm(ParComm* comm, const aligned_vector<int>& on_proc_col_to_new,
                const aligned_vector<int>& off_proc_col_to_new) 
            : CommPkg(comm->topology)
        {
            mpi_comm = comm->mpi_comm;
            bool comm_proc;
            int proc, start, end;
            int idx, new_idx;

            if (comm == NULL)
            {
                key = 0;
                return;
            }
            key = comm->key;

            init_off_proc_new(comm, off_proc_col_to_new);

            for (int i = 0; i < send_data->size_msgs; i++)
            {
                idx = send_data->indices[i];
                new_idx = on_proc_col_to_new[idx];
                if (new_idx != -1)
                {
                    send_data->indices[i] = new_idx;
                }
            }
        }


        void init_off_proc_new(ParComm* comm, const aligned_vector<int>& off_proc_col_to_new)
        {
            bool comm_proc, comm_idx;
            int proc, start, end;
            int idx, new_idx, ctr;
            int idx_start, idx_end;

            std::function<int(int, int)> compare_func = [](const int a, const int b)
            {
                if (b >= 0) return b;
                else return a;
            };
            comm->communicate_T(off_proc_col_to_new, 1, 1, 0, compare_func, -1);

            recv_data = comm->recv_data->copy(off_proc_col_to_new);

            send_data = new NonContigData();
            for (int i = 0; i < comm->send_data->num_msgs; i++)
            {
                comm_proc = false;
                proc = comm->send_data->procs[i];
                start = comm->send_data->indptr[i];
                end = comm->send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    if (comm->send_data->int_buffer[j] != -1)
                    {
                        comm_proc = true;
                        send_data->indices.emplace_back(comm->send_data->indices[j]);
                    }
                }
                if (comm_proc)
                {
                    send_data->procs.emplace_back(proc);
                    send_data->indptr.emplace_back(send_data->indices.size());
                }
            }
            send_data->num_msgs = send_data->procs.size();
            send_data->size_msgs = send_data->indices.size();
            send_data->finalize();

            
        }

        /**************************************************************
        *****   ParComm Class Destructor
        **************************************************************
        ***** 
        **************************************************************/
        ~ParComm()
        {
            delete send_data;
            delete recv_data;
        }

        // Standard Communication
        void init_double_comm(const double* values, const int block_size = 1,
                const int vblock_size = 1, const int vblock_offset = 0)
        {
            initialize(values, block_size, vblock_size, vblock_offset);
        }
        void init_int_comm(const int* values, const int block_size = 1,
                const int vblock_size = 1)
        {
            initialize(values);
        }
        aligned_vector<double>& complete_double_comm(const int block_size = 1,
                const int vblock_size = 1)
        {
            return complete<double>(block_size, vblock_size);
        }
        aligned_vector<int>& complete_int_comm(const int block_size = 1,
                const int vblock_size = 1)
        {
            //return complete<int>(block_size, vblock_size);
            return complete<int>(block_size);
        }
        template<typename T>
        aligned_vector<T>& communicate(const aligned_vector<T>& values,
                const int block_size = 1, const int vblock_size = 1,
                const int vblock_offset = 0)
        {
            return CommPkg::communicate(values.data(), block_size);
        }
        template<typename T>
        aligned_vector<T>& communicate(const T* values, const int block_size = 1,
                const int vblock_size = 1, const int vblock_offset = 0)
        {
            return CommPkg::communicate(values, block_size, vblock_size, vblock_offset);
        }

        template<typename T>
        void initialize(const T* values, const int block_size = 1,
                const int vblock_size = 1, const int vblock_offset = 0)
        {
            int start, end;
            int proc, pos, idx;
            
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
            if (profile) vec_t -= RAPtor_MPI_Wtime();
            send_data->send(values, key, mpi_comm, block_size, vblock_size, vblock_offset);
            recv_data->recv<T>(key, mpi_comm, block_size, vblock_size);
            if (profile) vec_t += RAPtor_MPI_Wtime();
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
        }

        template<typename T>
        aligned_vector<T>& complete(const int block_size = 1, const int vblock_size = 1)
        {
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
            if (profile) vec_t -= RAPtor_MPI_Wtime();
            send_data->waitall();
            recv_data->waitall();
            if (profile) vec_t += RAPtor_MPI_Wtime();
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
            key++;

            // Extract packed data to appropriate buffer
            aligned_vector<T>& buf = recv_data->get_buffer<T>();

            // Reorder the buffer if receiving block vector
            if (vblock_size > 1 && recv_data->procs.size() > 1)
            {
                aligned_vector<T> temp;
                int vec_size, start, end, vec_start, vec_end;
                int total_vec_size = buf.size() / vblock_size;
                for (int v = 0; v < vblock_size; v++)
                {
                    for (int i = 0; i < recv_data->num_msgs; i++)
                    {
                        start = recv_data->indptr[i];
                        end = recv_data->indptr[i+1];
                        vec_size = (end - start) * block_size;
                        start = start * block_size * vblock_size;
                        vec_start = start + v * vec_size;
                        vec_end = vec_start + vec_size;
                        for (int j = vec_start; j < vec_end; j++)
                        {
                            temp.emplace_back(buf[j]);
                        }
                    }
                }
                std::copy(temp.begin(), temp.end(), buf.begin());
            }

            return buf;
        }

        // Transpose Communication
        void init_double_comm_T(const double* values,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<double(double, double)> init_result_func = 
                    &sum_func<double, double>, 
                    double init_result_func_val = 0)
        {
            initialize_T(values, block_size, vblock_size, vblock_offset, init_result_func, init_result_func_val);
        }
        void init_int_comm_T(const int* values,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<int(int, int)> init_result_func = 
                    &sum_func<int, int>, 
                    int init_result_func_val = 0)
        {
            initialize_T(values, block_size, vblock_size, vblock_offset, init_result_func, init_result_func_val);
        }
        void complete_double_comm_T(aligned_vector<double>& result,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<double(double, double)> result_func = &sum_func<double, double>,
                std::function<double(double, double)> init_result_func = 
                    &sum_func<double, double>,
                    double init_result_func_val = 0)
        {
            complete_T<double>(result, block_size, vblock_size, vblock_offset, result_func, init_result_func, init_result_func_val);
        }
        void complete_double_comm_T(aligned_vector<int>& result,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<int(int, double)> result_func = &sum_func<double, int>,
                std::function<double(double, double)> init_result_func = 
                    &sum_func<double, double>,
                    double init_result_func_val = 0)
        {
            complete_T<double>(result, block_size, vblock_size, vblock_offset, result_func, init_result_func, init_result_func_val);
        }
        void complete_int_comm_T(aligned_vector<double>& result,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<double(double, int)> result_func = &sum_func<int, double>,
                std::function<int(int, int)> init_result_func = &sum_func<int, int>,
                int init_result_func_val = 0)
        {
            complete_T<int>(result, block_size, vblock_size, vblock_offset, result_func, init_result_func, init_result_func_val);
        }
        void complete_int_comm_T(aligned_vector<int>& result,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<int(int, int)> result_func = &sum_func<int, int>,
                std::function<int(int, int)> init_result_func = &sum_func<int, int>,
                int init_result_func_val = 0)
        {
            complete_T<int>(result, block_size, vblock_size, vblock_offset, result_func, init_result_func, init_result_func_val);
        }
        void complete_double_comm_T(const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<double(double, double)> init_result_func =
                &sum_func<double, double>, 
                double init_result_func_val = 0)
        {
            complete_T<double>(block_size, vblock_size, vblock_offset, init_result_func, init_result_func_val);
        }
        void complete_int_comm_T(const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<int(int, int)> init_result_func = &sum_func<int, int>,
                int init_result_func_val = 0)
        {
            complete_T<int>(block_size, vblock_size, vblock_offset, init_result_func, init_result_func_val);
        }
        template<typename T, typename U>
        void communicate_T(const aligned_vector<T>& values, aligned_vector<U>& result,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<U(U, T)> result_func = &sum_func<T, U>,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>, 
                T init_result_func_val = 0)
        {
            CommPkg::communicate_T(values.data(), result, block_size,
                    vblock_size, vblock_offset,
                    result_func, init_result_func, init_result_func_val);
        }
        template<typename T, typename U>
        void communicate_T(const T* values, aligned_vector<U>& result,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<U(U, T)> result_func = &sum_func<T, U>,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            CommPkg::communicate_T(values, result, block_size,
                    vblock_size, vblock_offset,
                    result_func, init_result_func, init_result_func_val);
        }
        template<typename T>
        void communicate_T(const aligned_vector<T>& values,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            CommPkg::communicate_T(values.data(), block_size, vblock_size, vblock_offset, init_result_func,
                    init_result_func_val);
        }
        template<typename T>
        void communicate_T(const T* values, const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            CommPkg::communicate_T(values, block_size, vblock_size, vblock_offset, init_result_func,
                    init_result_func_val);
        }

        template<typename T>
        void initialize_T(const T* values, const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>, 
                T init_result_func_val = 0)
        {
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
            if (profile) vec_t -= RAPtor_MPI_Wtime();
            recv_data->send(values, key, mpi_comm, block_size, vblock_size, vblock_offset,
                    init_result_func, init_result_func_val);
            send_data->recv<T>(key, mpi_comm, block_size, vblock_size);
            if (profile) vec_t += RAPtor_MPI_Wtime();
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
        }

        template<typename T, typename U>
        void complete_T(aligned_vector<U>& result, 
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<U(U, T)> result_func = &sum_func<T, U>,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            // TODO - dont need to copy into sendbuf first
            complete_T<T>(block_size, vblock_size, vblock_offset, init_result_func, init_result_func_val);

            int idx, pos;
            aligned_vector<T>& sendbuf = send_data->get_buffer<T>();

            int result_vec_offset = result.size() / vblock_size;
            int start, end;

            if (vblock_size > 1)
            {
                for (int i = 0; i < send_data->num_msgs; i++)
                {
                    start = send_data->indptr[i];
                    end = send_data->indptr[i+1];
                    for (int j = 0; j < (end - start); j++)
                    {
                        idx = send_data->indices[j + start] * block_size;
                        pos = j * block_size + (start * vblock_size);
                        for (int b = 0; b < block_size; b++)
                        {
                            for (int v = 0; v < vblock_size; v++)
                            {
                                result[v*result_vec_offset + idx + b]  = 
                                    result_func(result[v*result_vec_offset + idx + b],
                                    sendbuf[v*(end - start) + pos + b]);
                            }
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < send_data->size_msgs; i++)
                {
                    idx = send_data->indices[i] * block_size;
                    pos = i * block_size;
                    for (int b = 0; b < block_size; b++)
                    {
                        result[idx + b]  = result_func(result[idx + b], sendbuf[pos + b]);
                    }
                }
            }
        }

        template<typename T>
        void complete_T(const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
            if (profile) vec_t -= RAPtor_MPI_Wtime();
            send_data->waitall();
            recv_data->waitall();
            if (profile) vec_t += RAPtor_MPI_Wtime();
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
            key++;
            
            aligned_vector<T>& buf = send_data->get_buffer<T>();
        }

        // Conditional communication
        template <typename T>
        aligned_vector<T>& conditional_comm(
                const aligned_vector<T>& vals,  
                const aligned_vector<int>& states, 
                const aligned_vector<int>& off_proc_states,
                std::function<bool(int)> compare_func,
                const int block_size = 1,
                const int vblock_size = 1)
        {
            int ctr, n_sends, n_recvs;
            int key = 325493;
            bool comparison;
            
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
            if (profile) vec_t -= RAPtor_MPI_Wtime();
            send_data->send(vals.data(), key, mpi_comm, states, compare_func, &n_sends, block_size);
            recv_data->recv<T>(key, mpi_comm, off_proc_states, 
                    compare_func, &ctr, &n_recvs, block_size);

            send_data->waitall(n_sends);
            recv_data->waitall(n_recvs);
            if (profile) vec_t += RAPtor_MPI_Wtime();
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);

            aligned_vector<T>& recvbuf = recv_data->get_buffer<T>();

            ctr--;
            for (int i = recv_data->size_msgs - 1; i >= 0; i--)
            {
                int idx = i * block_size;
                comparison = false;
                for (int j = 0; j < block_size; j++)
                {
                    if (compare_func(off_proc_states[idx+j]))
                    {
                        comparison = true;
                        break;
                    }
                }
                if (comparison)
                {
                    for (int j = block_size - 1; j >= 0; j--)
                    {
                        recvbuf[idx+j] = recvbuf[ctr--];
                    }
                }
                else
                {
                    for (int j = block_size - 1; j >= 0; j--)
                    {
                        recvbuf[idx+j] = 0.0;
                    }
                }
            }

            return recvbuf;
        }

        template <typename T, typename U>
        void conditional_comm_T(const aligned_vector<T>& vals,  
                const aligned_vector<int>& states, 
                const aligned_vector<int>& off_proc_states,
                std::function<bool(int)> compare_func,
                aligned_vector<U>& result, 
                std::function<U(U, T)> result_func,
                const int block_size = 1,
                const int vblock_size = 1)
        {
            int idx, ctr;
            int n_sends, n_recvs;
            int key = 453246;
            bool comparison;

            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
            if (profile) vec_t -= RAPtor_MPI_Wtime();
            recv_data->send(vals.data(), key, mpi_comm, off_proc_states, compare_func,
                    &n_sends, block_size);
            send_data->recv<T>(key, mpi_comm, states, compare_func, &ctr, &n_recvs, block_size);
            
            recv_data->waitall(n_sends);
            send_data->waitall(n_recvs);
            if (profile) vec_t += RAPtor_MPI_Wtime();
            RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);

            aligned_vector<T>& sendbuf = send_data->get_buffer<T>();

            ctr = 0;
            for (int i = 0; i < send_data->size_msgs; i++)
            {
                idx = send_data->indices[i] * block_size;
                comparison = false;
                for (int j = 0; j < block_size; j++)
                {
                    if (compare_func(states[idx + j]))
                    {
                        comparison = true;
                        break;
                    }
                }
                if (comparison)
                {
                    for (int j = 0; j < block_size; j++)
                    {
                        result[idx + j] = result_func(result[idx + j], sendbuf[ctr++]);
                    }
                }
            }
        }


        // Matrix Communication
        CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true);
        CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true);
        void init_mat_comm(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true);
        void init_mat_comm(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true);
        CSRMatrix* complete_mat_comm(const int b_rows = 1, const int b_cols = 1, 
                const bool has_vals = true);

        CSRMatrix* communicate_T(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values, 
                const int n_result_rows, const int b_rows = 1, const int b_cols = 1, 
                const bool has_vals = true);
        CSRMatrix* communicate_T(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double*>& values, 
                const int n_result_rows, const int b_rows = 1, const int b_cols = 1, 
                const bool has_vals = true);
        void init_mat_comm_T(aligned_vector<char>& send_buffer, 
                const aligned_vector<int>& rowptr, const aligned_vector<int>& col_indices, 
                const aligned_vector<double>& values, const int b_rows = 1, 
                const int b_cols = 1, const bool has_vals = true) ;
        void init_mat_comm_T(aligned_vector<char>& send_buffer,
                const aligned_vector<int>& rowptr, const aligned_vector<int>& col_indices, 
                const aligned_vector<double*>& values, const int b_rows = 1, 
                const int b_cols = 1, const bool has_vals = true) ;
        CSRMatrix* complete_mat_comm_T(const int n_result_rows, 
                const int b_rows = 1, const int b_cols = 1,
                const bool has_vals = true) ;


        CSRMatrix* communicate(ParCSRMatrix* A, const bool has_vals = true)
        {
            return CommPkg::communicate(A, has_vals);
        }
        CSRMatrix* communicate(ParBSRMatrix* A, const bool has_vals = true)
        {
            return CommPkg::communicate(A, has_vals);
        }
        CSRMatrix* communicate(CSRMatrix* A, const bool has_vals = true)
        {
            return CommPkg::communicate(A, has_vals);
        }
        CSRMatrix* communicate_T(CSRMatrix* A, const bool has_vals = true)
        {
            return CommPkg::communicate_T(A, has_vals);
        }


        // Vector Communication
        aligned_vector<double>& communicate(ParVector& v, const int block_size = 1,
                const int vblock_size = 1, const int vblock_offset = 0)
        {
            return CommPkg::communicate(v, block_size);
        }
        void init_comm(ParVector& v, const int block_size = 1, const int vblock_size = 1)
        {
            CommPkg::init_comm(v, block_size, vblock_size);
        }

        // Helper Methods
        aligned_vector<double>& get_double_buffer()
        {
            return recv_data->buffer;
        }
        aligned_vector<int>& get_int_buffer()
        {
            return recv_data->int_buffer;
        }

        int key;
        NonContigData* send_data;
        CommData* recv_data;
        RAPtor_MPI_Comm mpi_comm;
    };



    /**************************************************************
    *****   TAPComm Class
    **************************************************************
    ***** This class constructs a topology-aware parallel communicator: 
    ***** which messages must be sent/recieved for matrix operations,
    ***** using topology-aware methods to limit the number and size
    ***** of inter-node messages
    *****
    ***** Attributes
    ***** -------------
    ***** local_S_par_comm : ParComm*
    *****    Parallel communication package for sending data that originates
    *****    on rank to other processes local to node, before inter-node
    *****    communication occurs.
    ***** local_R_par_comm : ParComm*
    *****    Parallel communication package for redistributing previously
    *****    received values (from inter-node communication step) to 
    *****    processes local to rank which need said values
    ***** local_L_par_comm : ParComm* 
    *****    Parallel communication package for communicating values
    *****    that both originate and have a final destination on node
    *****    (fully intra-node communication)
    ***** global_par_comm : ParComm*
    *****    Parallel communication package for sole inter-node step.
    ***** buffer : Vector
    *****    Combination of local_L_par_comm and local_R_par_comm
    *****    recv buffers, ordered to match off_proc_column_map
    ***** Partition* partition
    *****    Partition, holding information about topology
    **************************************************************/
    class TAPComm : public CommPkg
    {
        public:

        TAPComm(Partition* partition, bool form_S = true, ParComm* L_comm = NULL) : CommPkg(partition)
        {
            if (form_S)
            {
                local_S_par_comm = new ParComm(partition, 2345, partition->topology->local_comm,
                        new DuplicateData());
            }
            else local_S_par_comm = NULL;

            local_R_par_comm = new ParComm(partition, 3456, partition->topology->local_comm,
                    new NonContigData());
            global_par_comm = new ParComm(partition, 5678, RAPtor_MPI_COMM_WORLD,
                    new DuplicateData());

            if (L_comm)
            {
                local_L_par_comm = L_comm;
                local_L_par_comm->num_shared++;
            }
            else
            {
                local_L_par_comm = new ParComm(partition, 4567, partition->topology->local_comm,
                        new NonContigData());
            }
        }


        /**************************************************************
        *****   TAPComm Class Constructor
        **************************************************************
        ***** Initializes a TAPComm for a matrix without contiguous
        ***** row-wise partitions across processes.  Instead, each
        ***** process holds a random assortment of rows. 
        *****
        ***** Parameters
        ***** -------------
        ***** off_proc_column_map : aligned_vector<int>&
        *****    Maps local off_proc columns indices to global
        ***** global_num_cols : int
        *****    Number of global columns in matrix
        ***** local_num_cols : int
        *****    Number of columns local to rank
        **************************************************************/
        TAPComm(Partition* partition, 
                const aligned_vector<int>& off_proc_column_map,
                bool form_S = true,
                RAPtor_MPI_Comm comm = RAPtor_MPI_COMM_WORLD,
                int msg_cap = 0)
                : CommPkg(partition)
        {
            if (msg_cap > 0)
            {
                init_tap_comm_optimal(partition, off_proc_column_map, comm, msg_cap);
            }
            else
            {
                if (form_S)
                {
                    init_tap_comm(partition, off_proc_column_map, comm);
                }
                else
                {
                    init_tap_comm_simple(partition, off_proc_column_map, comm);
                }
            }
        }

        TAPComm(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                const aligned_vector<int>& on_proc_column_map,
                bool form_S = true,
                RAPtor_MPI_Comm comm = RAPtor_MPI_COMM_WORLD,
                int msg_cap = 0)
                : CommPkg(partition)
        {
            aligned_vector<int> on_proc_to_new;
            int on_proc_num_cols = on_proc_column_map.size();
            if (partition->local_num_cols)
            {
                on_proc_to_new.resize(partition->local_num_cols);
                for (int i = 0; i < on_proc_num_cols; i++)
                {
                    on_proc_to_new[on_proc_column_map[i] - partition->first_local_col] = i;
                }
            }

            if (msg_cap > 0)
            {
                init_tap_comm_optimal(partition, off_proc_column_map, comm, msg_cap);
                for (aligned_vector<int>::iterator it = global_par_comm->send_data->indices.begin();
                        it != global_par_comm->send_data->indices.end(); ++it)
                {
                    *it = on_proc_to_new[*it];
                }
            }
            if (form_S)
            {
                init_tap_comm(partition, off_proc_column_map, comm);

                for (aligned_vector<int>::iterator it = local_S_par_comm->send_data->indices.begin();
                        it != local_S_par_comm->send_data->indices.end(); ++it)
                {
                    *it = on_proc_to_new[*it];
                }
            }
            else
            {
                init_tap_comm_simple(partition, off_proc_column_map, comm);

                for (aligned_vector<int>::iterator it = global_par_comm->send_data->indices.begin();
                        it != global_par_comm->send_data->indices.end(); ++it)
                {
                    *it = on_proc_to_new[*it];
                }
            }

            for (aligned_vector<int>::iterator it = local_L_par_comm->send_data->indices.begin();
                    it != local_L_par_comm->send_data->indices.end(); ++it)
            {
                *it = on_proc_to_new[*it];
            }
        }

        /**************************************************************
        *****   TAPComm Class Constructor
        **************************************************************
        ***** Create topology-aware communication class from 
        ***** original communication package (which processes rank
        ***** communication which, and what is sent to / recv from
        ***** each process.
        *****
        ***** Parameters
        ***** -------------
        ***** orig_comm : ParComm*
        *****    Existing standard communication package from which
        *****    to form topology-aware communicator
        **************************************************************/
        TAPComm(TAPComm* tap_comm) : CommPkg(tap_comm->topology)
        {
            if (tap_comm->local_S_par_comm)
            {
                local_S_par_comm = new ParComm(tap_comm->local_S_par_comm);
            }
            else local_S_par_comm = NULL;

            global_par_comm = new ParComm(tap_comm->global_par_comm);
            local_R_par_comm = new ParComm(tap_comm->local_R_par_comm);
            local_L_par_comm = new ParComm(tap_comm->local_L_par_comm);

            recv_size = tap_comm->recv_size;
            if (recv_size)
            {
                buffer.resize(recv_size);
                int_buffer.resize(recv_size);
            }
        }

        TAPComm(TAPComm* tap_comm, const aligned_vector<int>& off_proc_col_to_new, 
                ParComm* local_L = NULL) : CommPkg(tap_comm->topology)
        {
            init_off_proc_new(tap_comm, off_proc_col_to_new, local_L);
        }

        TAPComm(TAPComm* tap_comm, const aligned_vector<int>& on_proc_col_to_new,
                const aligned_vector<int>& off_proc_col_to_new, 
                ParComm* local_L = NULL) : CommPkg(tap_comm->topology)
        {
            int idx;

            init_off_proc_new(tap_comm, off_proc_col_to_new, local_L);

            if (!local_L)
            {
                for (int i = 0; i < local_L_par_comm->send_data->size_msgs; i++)
                {
                    idx = local_L_par_comm->send_data->indices[i];
                    local_L_par_comm->send_data->indices[i] = on_proc_col_to_new[idx];
                }
            }

            if (local_S_par_comm)
            {
                for (int i = 0; i < local_S_par_comm->send_data->size_msgs; i++)
                {
                    idx = local_S_par_comm->send_data->indices[i];
                    local_S_par_comm->send_data->indices[i] = on_proc_col_to_new[idx];
                }
            }
            else
            {
                for (int i = 0; i < global_par_comm->send_data->size_msgs; i++)
                {
                    idx = global_par_comm->send_data->indices[i];
                    global_par_comm->send_data->indices[i] = on_proc_col_to_new[idx];
                }
            }
        }


        void init_off_proc_new(TAPComm* tap_comm, const aligned_vector<int>& off_proc_col_to_new,
                ParComm* local_L = NULL)
        {
            int idx, ctr;
            int start, end;

            DuplicateData* global_recv = (DuplicateData*) tap_comm->global_par_comm->recv_data;

            if (local_L)
            {
                local_L_par_comm = local_L;
                local_L_par_comm->num_shared++;
            }
            else
            {
                local_L_par_comm = new ParComm(tap_comm->local_L_par_comm, off_proc_col_to_new);
            }
            local_R_par_comm = new ParComm(tap_comm->local_R_par_comm, off_proc_col_to_new);

            // Create global par comm / update R send indices
            aligned_vector<int>& local_R_int_buffer = 
                tap_comm->local_R_par_comm->send_data->get_buffer<int>();
            aligned_vector<int>& global_int_buffer = 
                tap_comm->global_par_comm->send_data->get_buffer<int>();

            aligned_vector<int> G_to_new(tap_comm->global_par_comm->recv_data->size_msgs, -1);
            ctr = 0;
            for (int i = 0; i < global_recv->size_msgs; i++)
            {
                start = global_recv->indptr_T[i];
                end = global_recv->indptr_T[i+1];
                for (int j = start; j < end; j++)
                {
                    idx = global_recv->indices[j];
                    if (local_R_int_buffer[idx] != -1)
                    {
                        G_to_new[i] = ctr++;
                        break;
                    }
                }
            }
            for (aligned_vector<int>::iterator it = local_R_par_comm->send_data->indices.begin();
                    it != local_R_par_comm->send_data->indices.end(); ++it)
            {
                *it = G_to_new[*it];
            }
            idx = 0;
            for (aligned_vector<int>::iterator it = local_R_int_buffer.begin();
                    it != local_R_int_buffer.end(); ++it)
            {
                if (*it != -1) *it = idx++;
            }

            global_par_comm = new ParComm(tap_comm->global_par_comm, 
                    local_R_int_buffer);


            // create local S / update global send indices
            if (tap_comm->local_S_par_comm)
            {
                DuplicateData* local_S_recv = (DuplicateData*) tap_comm->local_S_par_comm->recv_data;
                aligned_vector<int> S_to_new(tap_comm->local_S_par_comm->recv_data->size_msgs, -1);
                ctr = 0;
                for (int i = 0; i < local_S_recv->size_msgs; i++)
                {
                    start = local_S_recv->indptr_T[i];
                    end = local_S_recv->indptr_T[i+1];
                    for (int j = start; j < end; j++)
                    {
                        idx = local_S_recv->indices[j];
                        if (global_int_buffer[idx] != -1)
                        {
                            S_to_new[i] = ctr++;
                            break;
                        }
                    }
                }
                for (aligned_vector<int>::iterator it = global_par_comm->send_data->indices.begin();
                        it != global_par_comm->send_data->indices.end(); ++it)
                {
                    *it = S_to_new[*it];
                }
                idx = 0;
                for (aligned_vector<int>::iterator it = global_int_buffer.begin(); 
                        it != global_int_buffer.end(); ++it)
                {
                    if (*it != -1) *it = idx++;
                }

                local_S_par_comm = new ParComm(tap_comm->local_S_par_comm,
                        global_int_buffer);
            }
            else local_S_par_comm = NULL;

            // Determine size of final recvs (should be equal to 
            // number of off_proc cols)
            recv_size = local_R_par_comm->recv_data->size_msgs +
                local_L_par_comm->recv_data->size_msgs;
            if (recv_size)
            {
                // Want a single recv buffer local_R and local_L par_comms
                buffer.resize(recv_size);
                int_buffer.resize(recv_size);
            }        
        }

        /**************************************************************
        *****   ParComm Class Destructor
        **************************************************************
        ***** 
        **************************************************************/
        ~TAPComm()
        {
            if (global_par_comm)
                global_par_comm->delete_comm();
            if (local_S_par_comm)
                local_S_par_comm->delete_comm();
            if (local_R_par_comm)
                local_R_par_comm->delete_comm();
            if (local_L_par_comm)
                local_L_par_comm->delete_comm();
        }

        void init_tap_comm(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                RAPtor_MPI_Comm comm)
        {
            // Get RAPtor_MPI Information
            int rank, num_procs;
            RAPtor_MPI_Comm_rank(comm, &rank);
            RAPtor_MPI_Comm_size(comm, &num_procs);

            // Initialize class variables
            local_S_par_comm = new ParComm(partition, 2345, partition->topology->local_comm, 
                    new DuplicateData());
            local_R_par_comm = new ParComm(partition, 3456, partition->topology->local_comm,
                    new NonContigData());
            local_L_par_comm = new ParComm(partition, 4567, partition->topology->local_comm,
                    new NonContigData());
            global_par_comm = new ParComm(partition, 5678, comm, new DuplicateData());

            // Initialize Variables
            int idx;
            aligned_vector<int> off_proc_col_to_proc;
            aligned_vector<int> on_node_column_map;
            aligned_vector<int> on_node_col_to_proc;
            aligned_vector<int> off_node_column_map;
            aligned_vector<int> off_node_col_to_node;
            aligned_vector<int> on_node_to_off_proc;
            aligned_vector<int> off_node_to_off_proc;
            aligned_vector<int> recv_nodes;
            aligned_vector<int> orig_procs;
            aligned_vector<int> node_to_local_proc;

            // Find process on which vector value associated with each column is
            // stored
            partition->form_col_to_proc(off_proc_column_map, off_proc_col_to_proc);

            // Partition off_proc cols into on_node and off_node
            split_off_proc_cols(off_proc_column_map, off_proc_col_to_proc,
                   on_node_column_map, on_node_col_to_proc, on_node_to_off_proc,
                   off_node_column_map, off_node_col_to_node, off_node_to_off_proc);

            // Gather all nodes with which any local process must communication
            form_local_R_par_comm(off_node_column_map, off_node_col_to_node, 
                    orig_procs);

            // Find global processes with which rank communications
            form_global_par_comm(orig_procs);

            // Form local_S_par_comm: initial distribution of values among local
            // processes, before inter-node communication
            form_local_S_par_comm(orig_procs);

            // Adjust send indices (currently global vector indices) to be index 
            // of global vector value from previous recv
            adjust_send_indices(partition->first_local_col);

            // Form local_L_par_comm: fully local communication (origin and
            // destination processes both local to node)
            form_local_L_par_comm(on_node_column_map, on_node_col_to_proc,
                    partition->first_local_col);

            // Determine size of final recvs (should be equal to 
            // number of off_proc cols)
            update_recv(on_node_to_off_proc, off_node_to_off_proc);
        }

        void init_tap_comm_simple(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                RAPtor_MPI_Comm comm)
        {
            // Get RAPtor_MPI Information
            int rank, num_procs;
            RAPtor_MPI_Comm_rank(comm, &rank);
            RAPtor_MPI_Comm_size(comm, &num_procs);

            // Initialize class variables
            local_S_par_comm = NULL;
            local_R_par_comm = new ParComm(partition, 3456, partition->topology->local_comm, 
                    new NonContigData());
            local_L_par_comm = new ParComm(partition, 4567, partition->topology->local_comm,
                    new NonContigData());
            global_par_comm = new ParComm(partition, 5678, comm, new DuplicateData());

            // Initialize Variables
            int idx;
            aligned_vector<int> off_proc_col_to_proc;
            aligned_vector<int> on_node_column_map;
            aligned_vector<int> on_node_col_to_proc;
            aligned_vector<int> off_node_column_map;
            aligned_vector<int> off_node_col_to_proc;
            aligned_vector<int> on_node_to_off_proc;
            aligned_vector<int> off_node_to_off_proc;

            // Find process on which vector value associated with each column is
            // stored
            partition->form_col_to_proc(off_proc_column_map, off_proc_col_to_proc);

            // Partition off_proc cols into on_node and off_node
            split_off_proc_cols(off_proc_column_map, off_proc_col_to_proc,
                   on_node_column_map, on_node_col_to_proc, on_node_to_off_proc,
                   off_node_column_map, off_node_col_to_proc, off_node_to_off_proc);

            // Form local recv communicator.  Will recv from local rank
            // corresponding to global rank on which data originates.  E.g. if
            // data is on rank r = (p, n), and my rank is s = (q, m), I will
            // recv data from (p, m).
            form_simple_R_par_comm(off_node_column_map, off_node_col_to_proc);

            // Form global par comm.. Will recv from proc on which data
            // originates
            form_simple_global_comm(off_node_col_to_proc);

            // Adjust send indices (currently global vector indices) to be
            // index of global vector value from previous recv (only updating
            // local_R to match position in global)
            adjust_send_indices(partition->first_local_col);

            // Form local_L_par_comm: fully local communication (origin and
            // destination processes both local to node)
            form_local_L_par_comm(on_node_column_map, on_node_col_to_proc,
                    partition->first_local_col);

            // Determine size of final recvs (should be equal to 
            // number of off_proc cols)
            update_recv(on_node_to_off_proc, off_node_to_off_proc);

        }
        
        void init_tap_comm_optimal(Partition* partition,
                const aligned_vector<int>& off_proc_column_map,
                RAPtor_MPI_Comm comm, int msg_cap)
        {
            // Hard Coding Cutoffs for now
            int short_cutoff = 500;
            int eager_cutoff = 8000;

            // Get RAPtor_MPI Information
            int rank, num_procs;
            RAPtor_MPI_Comm_rank(comm, &rank);
            RAPtor_MPI_Comm_size(comm, &num_procs);

            // Initialize class variables
            local_S_par_comm = NULL;
            local_R_par_comm = new ParComm(partition, 3456, partition->topology->local_comm, 
                    new NonContigData());
            local_L_par_comm = new ParComm(partition, 4567, partition->topology->local_comm,
                    new NonContigData());
            global_par_comm = new ParComm(partition, 5678, comm, new DuplicateData());

            // Initialize Variables
            int idx;
            aligned_vector<int> off_proc_col_to_proc;
            aligned_vector<int> on_node_column_map;
            aligned_vector<int> on_node_col_to_proc;
            aligned_vector<int> off_node_column_map;
            aligned_vector<int> off_node_col_to_proc;
            aligned_vector<int> on_node_to_off_proc;
            aligned_vector<int> off_node_to_off_proc;

            // Find process on which vector value associated with each column is
            // stored
            partition->form_col_to_proc(off_proc_column_map, off_proc_col_to_proc);

            // Partition off_proc cols into on_node and off_node
            split_off_proc_cols(off_proc_column_map, off_proc_col_to_proc,
                   on_node_column_map, on_node_col_to_proc, on_node_to_off_proc,
                   off_node_column_map, off_node_col_to_proc, off_node_to_off_proc);

            for (int i =0; i < num_procs; i++)
            {
                if (i == rank)
                {
                    printf("%d off_node_col_to_proc ", rank);
                    for (int j = 0; j < off_node_col_to_proc.size(); j++)
                    {
                        printf("%d ", off_node_col_to_proc[j]);   
                    }
                    printf("\n"); 
                }
                fflush(stdout);
                RAPtor_MPI_Barrier(RAPtor_MPI_COMM_WORLD);
            }

            // Form local recv communicator.  Will recv from local rank
            // corresponding to global rank on which data originates.  E.g. if
            // data is on rank r = (p, n), and my rank is s = (q, m), I will
            // recv data from (p, m).
            form_optimal_R_par_comm(off_node_column_map, off_node_col_to_proc);

            // Rebalance R_par_comm -- conglomerating some of the messages

            // Form global par comm.. Will recv from proc on which data
            // originates
            form_simple_global_comm(off_node_col_to_proc);

            // Adjust send indices (currently global vector indices) to be
            // index of global vector value from previous recv (only updating
            // local_R to match position in global)
            adjust_send_indices(partition->first_local_col);

            // Form local_L_par_comm: fully local communication (origin and
            // destination processes both local to node)
            form_local_L_par_comm(on_node_column_map, on_node_col_to_proc,
                    partition->first_local_col);

            // Determine size of final recvs (should be equal to 
            // number of off_proc cols)
            update_recv(on_node_to_off_proc, off_node_to_off_proc);
        }

        // Helper methods for forming TAPComm:
        void split_off_proc_cols(const aligned_vector<int>& off_proc_column_map,
                const aligned_vector<int>& off_proc_col_to_proc,
                aligned_vector<int>& on_node_column_map,
                aligned_vector<int>& on_node_col_to_proc,
                aligned_vector<int>& on_node_to_off_proc,
                aligned_vector<int>& off_node_column_map,
                aligned_vector<int>& off_node_col_to_node,
                aligned_vector<int>& off_node_to_off_proc);
        void form_local_R_par_comm(const aligned_vector<int>& off_node_column_map,
                const aligned_vector<int>& off_node_col_to_node,
                aligned_vector<int>& orig_procs);
        void form_global_par_comm(aligned_vector<int>& orig_procs);
        void form_local_S_par_comm(aligned_vector<int>& orig_procs);
        void adjust_send_indices(const int first_local_col);
        void form_local_L_par_comm(const aligned_vector<int>& on_node_column_map,
                const aligned_vector<int>& on_node_col_to_proc,
                const int first_local_col);
        void form_simple_R_par_comm(aligned_vector<int>& off_node_column_map,
                aligned_vector<int>& off_node_col_to_proc);
        void form_simple_global_comm(aligned_vector<int>& off_node_col_to_proc);
        void update_recv(const aligned_vector<int>& on_node_to_off_proc,
                const aligned_vector<int>& off_node_to_off_proc, bool update_L = true);
        void form_optimal_R_par_comm(aligned_vector<int>& off_node_column_map,
                aligned_vector<int>& off_node_col_to_proc);

        // Class Methods
        void init_double_comm(const double* values, const int block_size,
                const int vblock_size = 1, const int vblock_offset = 0)
        {
            if (vblock_size > 1) initialize(values, block_size, vblock_size, vblock_offset);
            else initialize(values, block_size);
        }
        void init_int_comm(const int* values, const int block_size, const int vblock_size = 1)
        {
            initialize(values, block_size);
        }
        aligned_vector<double>& complete_double_comm(const int block_size, const int vblock_size = 1)
        {
            return complete<double>(block_size, vblock_size);
        }
        aligned_vector<int>& complete_int_comm(const int block_size, const int vblock_size = 1)
        {
            return complete<int>(block_size);
        }
        
        template<typename T>
        aligned_vector<T>& communicate(const aligned_vector<T>& values, 
                const int block_size = 1, const int vblock_size = 1, const int vblock_offset = 0)
        {
            return CommPkg::communicate<T>(values.data(), block_size);
        }
        template<typename T>
        aligned_vector<T>& communicate(const T* values,
                const int block_size = 1, const int vblock_size = 1, const int vblock_offset = 0)
        {
            return CommPkg::communicate<T>(values, block_size);
        }

        template<typename T>
        void initialize(const T* values, const int block_size = 1, const int vblock_size = 1,
                const int vblock_offset = 0)
        {
            // Messages with origin and final destination on node
            // SEND AND RECV DATA BUFFERS CORRECT
            local_L_par_comm->communicate<T>(values, block_size, vblock_size, vblock_offset);

            if (local_S_par_comm)
            {
                // Initial redistribution among node
                // SEND AND RECV DATA BUFFERS CORRECT
                aligned_vector<T>& S_vals = local_S_par_comm->communicate<T>(values, block_size,
                        vblock_size, vblock_offset);

                // Begin inter-node communication
                // SEND AND RECV DATA BUFFERS CORRECT 
                int offset = S_vals.size() / vblock_size;
                global_par_comm->initialize(S_vals.data(), block_size, vblock_size, offset);
            }
            else
            {
                // MIGHT NEED TO RECALCULATE VBLOCK_OFFSET HERE?
                global_par_comm->initialize(values, block_size, vblock_size, vblock_offset);
            }
        }

        template<typename T>
        aligned_vector<T>& complete(const int block_size = 1, const int vblock_size = 1)
        {
            // Complete inter-node communication
            // G_VALS CORRECT
            aligned_vector<T>& G_vals = global_par_comm->complete<T>(block_size, vblock_size);

            int offset = G_vals.size() / vblock_size;
            // Redistributing recvd inter-node values
            local_R_par_comm->communicate<T>(G_vals.data(), block_size, vblock_size, offset);
            
            aligned_vector<T>& recvbuf = get_buffer<T>();

            aligned_vector<T>& R_recvbuf = local_R_par_comm->recv_data->get_buffer<T>();
            aligned_vector<T>& L_recvbuf = local_L_par_comm->recv_data->get_buffer<T>();

            if (recvbuf.size() < recv_size * block_size * vblock_size)
                recvbuf.resize(recv_size * block_size * vblock_size);

            // Add values from L_recv and R_recv to appropriate positions in 
            // Vector recv
            int idx, new_idx, pos;
            int R_recv_size = local_R_par_comm->recv_data->size_msgs;
            int L_recv_size = local_L_par_comm->recv_data->size_msgs;
            NonContigData* local_R_recv = (NonContigData*) local_R_par_comm->recv_data;
            NonContigData* local_L_recv = (NonContigData*) local_L_par_comm->recv_data;

            int v_offset = recv_size * block_size;

            for (int i = 0; i < R_recv_size; i++)
            {
                for (int v = 0; v < vblock_size; v++)
                {
                    pos = i * block_size + v * R_recv_size;
                    idx = local_R_recv->indices[i] * block_size + v * v_offset;
                    for (int j = 0; j < block_size; j++)
                    {
                        recvbuf[idx + j] = R_recvbuf[pos + j];
                    }
                }
            }

            for (int i = 0; i < L_recv_size; i++)
            {
                for (int v = 0; v < vblock_size; v++)
                {
                    pos = i * block_size + v * L_recv_size;
                    idx = local_L_recv->indices[i] * block_size + v * v_offset;
                    for (int j = 0; j < block_size; j++)
                    {
                        recvbuf[idx + j] = L_recvbuf[pos + j];
                    }
                }
            }

            return recvbuf;
        }


        // Transpose Communication
        void init_double_comm_T(const double* values,
                const int block_size,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<double(double, double)> init_result_func = 
                    &sum_func<double, double>,
                    double init_result_func_val = 0)
        {
            initialize_T(values, block_size, vblock_size, vblock_offset, init_result_func, init_result_func_val);
        }
        void init_int_comm_T(const int* values,
                const int block_size,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<int(int, int)> init_result_func = &sum_func<int, int>,
                int init_result_func_val = 0)
        {
            initialize_T(values, block_size, vblock_size, vblock_offset, init_result_func, init_result_func_val);
        }
        void complete_double_comm_T(aligned_vector<double>& result,
                const int block_size,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<double(double, double)> result_func = &sum_func<double, double>,
                std::function<double(double, double)> init_result_func = 
                    &sum_func<double, double>,
                    double init_result_func_val = 0)
        {
            complete_T<double>(result, block_size, vblock_size, vblock_offset, result_func, init_result_func, init_result_func_val);
        }        
        void complete_double_comm_T(aligned_vector<int>& result,
                const int block_size,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<int(int, double)> result_func = &sum_func<double, int>,
                std::function<double(double, double)> init_result_func = 
                    &sum_func<double, double>,
                    double init_result_func_val = 0)
        {
            complete_T<double>(result, block_size, vblock_size, vblock_offset, result_func, init_result_func, init_result_func_val);
        }
        void complete_int_comm_T(aligned_vector<double>& result,
                const int block_size,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<double(double, int)> result_func = &sum_func<int, double>,
                std::function<int(int, int)> init_result_func = &sum_func<int, int>,
                int init_result_func_val = 0)
        {
            complete_T<int>(result, block_size, vblock_size, vblock_offset, result_func, init_result_func, init_result_func_val);
        }
        void complete_int_comm_T(aligned_vector<int>& result,
                const int block_size,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<int(int, int)> result_func = &sum_func<int, int>,
                std::function<int(int, int)> init_result_func = &sum_func<int, int>,
                int init_result_func_val = 0)
        {
            complete_T<int>(result, block_size, vblock_size, vblock_offset, result_func, init_result_func, init_result_func_val);
        }

        void complete_double_comm_T(const int block_size,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<double(double, double)> init_result_func = 
                &sum_func<double, double>,
                double init_result_func_val = 0)
        {
            complete_T<double>(block_size, vblock_size, vblock_offset, init_result_func, init_result_func_val);
        }
        void complete_int_comm_T(const int block_size,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<int(int, int)> init_result_func = 
                    &sum_func<int, int>,
                int init_result_func_val = 0)
        {
            complete_T<int>(block_size, vblock_size, vblock_offset, init_result_func, init_result_func_val);
        }

        template<typename T, typename U>
        void communicate_T(const aligned_vector<T>& values, aligned_vector<U>& result,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<U(U, T)> result_func = &sum_func<T, U>,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            CommPkg::communicate_T(values.data(), result, block_size, vblock_size, vblock_offset,
                    result_func, init_result_func, init_result_func_val);
        }
        template<typename T, typename U>
        void communicate_T(const T* values, aligned_vector<U>& result,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<U(U, T)> result_func = &sum_func<T, U>,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            CommPkg::communicate_T(values, result, block_size, vblock_size, vblock_offset,
                    result_func, init_result_func, init_result_func_val);
        }
        template<typename T>
        void communicate_T(const aligned_vector<T>& values,
                const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            CommPkg::communicate_T(values.data(), block_size, vblock_size, vblock_offset,
                    init_result_func, init_result_func_val);
        }
        template<typename T>
        void communicate_T(const T* values, const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            CommPkg::communicate_T(values, block_size, vblock_size, vblock_offset,
                    init_result_func, init_result_func_val);
        }

        template<typename T>
        void initialize_T(const T* values, const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            int rank, num_procs;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

            int idx, start, end, buf_pos, pos;

            // Messages with origin and final destination on node
            // SEND AND RECV BUFFERS CORRECT
            local_L_par_comm->communicate_T(values, block_size, vblock_size, vblock_offset,
                    init_result_func, init_result_func_val);

            // Initial redistribution among node
            // SEND AND RECV BUFFERS CORRECT
            local_R_par_comm->communicate_T(values, block_size, vblock_size, vblock_offset,
                    init_result_func, init_result_func_val);
           
            // Begin inter-node communication 
            aligned_vector<T>& R_sendbuf = local_R_par_comm->send_data->get_buffer<T>();

            // REORDER R_SENDBUF RIGHT HERE AND THEN PASS TO GLOBAL_PAR_COMM_INIT_COMM_T
            // Reorder R_Sendbuf to pass to global_par_comm init_comm_T
            // THIS MAY NEED TO BE UPDATED FOR BLOCK VECTOR + BLOCK MATRIX
            if (vblock_size > 1)
            {
                aligned_vector<T> R_sendbuf_reordered(R_sendbuf.size());
                int offset = R_sendbuf.size() / vblock_size;
                for (int i = 0; i < local_R_par_comm->send_data->num_msgs; i++)
                {
                    start = local_R_par_comm->send_data->indptr[i];        
                    end = local_R_par_comm->send_data->indptr[i+1];
                    buf_pos = start * block_size * vblock_size;
                    for (int j = start; j < end; j++)
                    {
                        for (int v = 0; v < vblock_size; v++)
                        {
                            for (int k = 0; k < block_size; k++)
                            {
                                R_sendbuf_reordered[v*offset+j+k] = R_sendbuf[buf_pos+v*(end-start)+k];
                            }
                        }
                        buf_pos++;
                    }
                }

                // BUFFER BEING PASSED IN ALREADY REORDERED
                global_par_comm->init_comm_T(R_sendbuf_reordered, block_size, vblock_size, offset,
                        init_result_func, init_result_func_val);
            }
            else
            {
                global_par_comm->init_comm_T(R_sendbuf, block_size, vblock_size, vblock_offset,
                        init_result_func, init_result_func_val);
            }
           
        }

        template<typename T, typename U>
        void complete_T(aligned_vector<U>& result, const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<U(U, T)> result_func = &sum_func<T, U>,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            // NEED TO CHECK WHICH SEND/ RECV LOOP IN HERE ISN'T RUNNING THROUGH ALL 3 VECTORS 
            int rank, num_procs;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
            complete_T<T>(block_size, vblock_size, vblock_offset, init_result_func, init_result_func_val);

            /*MPI_Barrier(MPI_COMM_WORLD);
            fflush(stdout);
            printf("after complete_T called vblock_size %d vblock_offset %d---------------\n", vblock_size, vblock_offset);
            MPI_Barrier(MPI_COMM_WORLD);*/

            int idx, pos, start, end;
            aligned_vector<T>& L_sendbuf = local_L_par_comm->send_data->get_buffer<T>();
            
            int v_offset = result.size() / vblock_size;
            
            if (vblock_size > 1)
            {
                for (int i = 0; i < local_L_par_comm->send_data->num_msgs; i++)
                {
                    start = local_L_par_comm->send_data->indptr[i];
                    end = local_L_par_comm->send_data->indptr[i+1];
                    for (int j = 0; j < (end-start); j++)
                    {
                        idx = local_L_par_comm->send_data->indices[j+start] * block_size;
                        pos = j * block_size + (start * vblock_size);
                        for (int b = 0; b < block_size; b++)
                        {
                            for (int v = 0; v < vblock_size; v++)
                            {
                                result[v*v_offset + idx + b] = 
                                        result_func(result[v*v_offset + idx + b],
                                        L_sendbuf[v*(end-start) + pos + b]);
                            }
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < local_L_par_comm->send_data->size_msgs; i++)
                {
                    idx = local_L_par_comm->send_data->indices[i] * block_size;
                    pos = i * block_size;
                    for (int b = 0; b < block_size; b++)
                    {
                        result[idx + b] = result_func(result[idx + b], L_sendbuf[pos + b]);
                    }
                }
            }

            if (local_S_par_comm)
            {
                aligned_vector<T>& S_sendbuf = local_S_par_comm->send_data->get_buffer<T>();

                if (vblock_size > 1)
                {
                    for (int i = 0; i < local_S_par_comm->send_data->num_msgs; i++)
                    {
                        start = local_S_par_comm->send_data->indptr[i];
                        end = local_S_par_comm->send_data->indptr[i+1];
                        for (int j = 0; j < (end-start); j++)
                        {
                            idx = local_S_par_comm->send_data->indices[j+start] * block_size;
                            //pos = j * block_size + (start * vblock_size);
                            pos = j * block_size * vblock_size;
                            for (int b = 0; b < block_size; b++)
                            {
                                for (int v = 0; v < vblock_size; v++)
                                {
                                    result[v*v_offset + idx + b] =
                                            result_func(result[v*v_offset + idx + b],
                                            S_sendbuf[v + pos + b]);
                                            //S_sendbuf[v*(end-start) + pos + b]);
                                }
                            }
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < local_S_par_comm->send_data->size_msgs; i++)
                    {
                        idx = local_S_par_comm->send_data->indices[i] * block_size;
                        pos = i * block_size;
                        for (int b = 0; b < block_size; b++)
                        {
                            result[idx + b] = result_func(result[idx + b], S_sendbuf[pos + b]);
                        }
                    }
                }
            }
            else
            {
                aligned_vector<T>& G_sendbuf = global_par_comm->send_data->get_buffer<T>();

                /*for (int p = 0; p < num_procs; p++)
                {
                    if (rank == p)
                    {
                        printf("%d G_sendbuf ", rank);
                        for (int i = 0; i < G_sendbuf.size(); i++)
                        {
                            printf("%e ", G_sendbuf[i]);
                        }
                        printf("\n");
                        fflush(stdout);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }*/

                if (vblock_size > 1)
                {
                    for (int i = 0; i < global_par_comm->send_data->num_msgs; i++)
                    {
                        start = global_par_comm->send_data->indptr[i];
                        end = global_par_comm->send_data->indptr[i+1];
                        for (int j = 0; j < (end-start); j++)
                        {
                            idx = global_par_comm->send_data->indices[j+start] * block_size;
                            //pos = j * block_size + (start * vblock_size);
                            pos = j * block_size * vblock_size;
                            for (int b = 0; b < block_size; b++)
                            {
                                for (int v = 0; v < vblock_size; v++)
                                {
                                    //printf("%d pos %d idx %d result[%d] += G_sendbuf[%d] %e + %e\n", rank, pos, idx, v*v_offset+idx+b, v+pos+b, result[v*v_offset+idx+b], G_sendbuf[v+pos+b]);
                                    result[v*v_offset + idx + b] =
                                            result_func(result[v*v_offset + idx + b],
                                            G_sendbuf[pos + b + v]);
                                            //G_sendbuf[v*(end-start) + pos + b]);
                                }
                            }
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < global_par_comm->send_data->size_msgs; i++)
                    {
                        idx = global_par_comm->send_data->indices[i] * block_size;
                        pos = i * block_size;
                        for (int b = 0; b < block_size; b++)
                        {
                            //printf("%d result[%d] += G_sendbuf[%d] %e + %e\n", rank, idx+b, pos+b, result[idx+b], G_sendbuf[pos+b]);
                            result[idx + b] = result_func(result[idx + b], G_sendbuf[pos + b]);
                        }
                    }
                }
                
                /*for (int p = 0; p < num_procs; p++)
                {
                    if (rank == p)
                    {
                        printf("%d result ", rank);
                        for (int i = 0; i < result.size(); i++)
                        {
                            printf("%e ", result[i]);
                        }
                        printf("\n");
                        fflush(stdout);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }*/
            }
        }
        template<typename T>
        void complete_T(const int block_size = 1,
                const int vblock_size = 1,
                const int vblock_offset = 0,
                std::function<T(T, T)> init_result_func = &sum_func<T, T>,
                T init_result_func_val = 0)
        {
            /*int rank, num_procs;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
            printf("%d this complete_T called first\n", rank);
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);*/
            
            // Complete inter-node communication
            // NEED TO RECALCULATE VBLOCK_OFFSET BASED ON SEND OR RECV BUF SIZE???
            global_par_comm->complete_comm_T<T>(block_size, vblock_size, vblock_offset,
                    init_result_func, init_result_func_val);
            
            /*fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD); 
            if (rank == 0) printf("after global par comm complete_comm_T\n");
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD); 
            for (int p = 0; p < num_procs; p++)
            {
                if (rank == p)
                {
                    if (global_par_comm->recv_data)
                    {
                        printf("%d global recv buf size %d\n", rank, global_par_comm->recv_data->buffer.size());
                        aligned_vector<T>& buf = global_par_comm->recv_data->get_buffer<T>();
                        printf("%d global recv_data ", rank);
                        for (int i = 0; i < buf.size(); i++)
                        {
                            printf("%e ", buf[i]);
                        }
                        printf("\n");
                    }
                    fflush(stdout);
                    if (global_par_comm->send_data)
                    {
                        printf("%d global send buf size %d\n", rank, global_par_comm->send_data->buffer.size());
                        aligned_vector<T>& buf = global_par_comm->send_data->get_buffer<T>();
                        printf("%d global send_data ", rank);
                        for (int i = 0; i < buf.size(); i++)
                        {
                            printf("%e ", buf[i]);
                        }
                        printf("\n");
                    }
                    fflush(stdout);
                }
                MPI_Barrier(MPI_COMM_WORLD); 
            }

            MPI_Barrier(MPI_COMM_WORLD);
            printf("%d After global par comm complete_comm_T\n", rank);
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);*/
    
            if (local_S_par_comm)
            {
                // THIS IS THE CALL THAT NEEDS THE VBLOCK_OFFSET RESET???
                aligned_vector<T>& G_sendbuf = global_par_comm->send_data->get_buffer<T>();
                int v_offset = G_sendbuf.size() / vblock_size;
                // REORDER SEND BUF SO THAT VECTORS ARE ORDERED V0, V1, V2.... AGAIN
                aligned_vector<T> G_sendbuf_reordered(G_sendbuf.size());
                int offset = G_sendbuf.size() / vblock_size;
                for (int v = 0; v < vblock_size; v++)
                {
                    for (int i = 0; i < offset; i++)
                    {
                        G_sendbuf_reordered[v*offset + i] = G_sendbuf[i*vblock_size + v];
                    }
                }

                local_S_par_comm->communicate_T(G_sendbuf_reordered, block_size, vblock_size, v_offset,
                        init_result_func, init_result_func_val);
            
                /*for (int p = 0; p < num_procs; p++)
                {
                    if (rank == p)
                    {
                        if (global_par_comm->send_data)
                        {
                            printf("%d G_sendbuf size %d\n", rank, G_sendbuf.size());
                            printf("%d G_sendbuf ", rank);
                            for (int i = 0; i < G_sendbuf.size(); i++)
                            {
                                printf("%e ", G_sendbuf[i]);
                            }
                            printf("\n");
                        }
                        fflush(stdout);
                        if (local_S_par_comm->recv_data)
                        {
                            printf("%d local_S_par_comm recv buf size %d\n", rank, local_S_par_comm->recv_data->buffer.size());
                            aligned_vector<T>& buf = local_S_par_comm->recv_data->get_buffer<T>();
                            printf("%d local_S_par_comm recv_data ", rank);
                            for (int i = 0; i < buf.size(); i++)
                            {
                                printf("%e ", buf[i]);
                            }
                            printf("\n");
                        }
                        fflush(stdout);
                        if (local_S_par_comm->send_data)
                        {
                            printf("%d local_S_par_comm send buf size %d\n", rank, local_S_par_comm->send_data->buffer.size());
                            aligned_vector<T>& buf = local_S_par_comm->send_data->get_buffer<T>();
                            printf("%d local_S_par_comm send_data ", rank);
                            for (int i = 0; i < buf.size(); i++)
                            {
                                printf("%e ", buf[i]);
                            }
                            printf("\n");
                        }
                        fflush(stdout);
                    }
                    MPI_Barrier(MPI_COMM_WORLD); 
                }*/

            }
        }


        // Matrix Communication
        CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true);
        CSRMatrix* communicate(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true);
        void init_mat_comm(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true);
        void init_mat_comm(aligned_vector<char>& send_buffer, const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double*>& values,
                const int b_rows = 1, const int b_cols = 1, const bool has_vals = true);
        CSRMatrix* complete_mat_comm(const int b_rows = 1, const int b_cols = 1, 
                const bool has_vals = true);

        CSRMatrix* communicate_T(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double>& values, 
                const int n_result_rows, const int b_rows = 1, const int b_cols = 1, 
                const bool has_vals = true);
        CSRMatrix* communicate_T(const aligned_vector<int>& rowptr, 
                const aligned_vector<int>& col_indices, const aligned_vector<double*>& values, 
                const int n_result_rows, const int b_rows = 1, const int b_cols = 1, 
                const bool has_vals = true);
        void init_mat_comm_T(aligned_vector<char>& send_buffer, 
                const aligned_vector<int>& rowptr, const aligned_vector<int>& col_indices, 
                const aligned_vector<double>& values, const int b_rows = 1, 
                const int b_cols = 1, const bool has_vals = true) ;
        void init_mat_comm_T(aligned_vector<char>& send_buffer,
                const aligned_vector<int>& rowptr, const aligned_vector<int>& col_indices, 
                const aligned_vector<double*>& values, const int b_rows = 1, 
                const int b_cols = 1, const bool has_vals = true) ;
        CSRMatrix* complete_mat_comm_T(const int n_result_rows, 
                const int b_rows = 1, const int b_cols = 1,
                const bool has_vals = true);

        CSRMatrix* communicate(ParCSRMatrix* A, const bool has_vals = true)
        {
            return CommPkg::communicate(A, has_vals);
        }
        CSRMatrix* communicate(ParBSRMatrix* A, const bool has_vals = true)
        {
            return CommPkg::communicate(A, has_vals);
        }
        CSRMatrix* communicate(CSRMatrix* A, const bool has_vals = true)
        {
            return CommPkg::communicate(A, has_vals);
        }
        CSRMatrix* communicate_T(CSRMatrix* A, const bool has_vals = true)
        {
            return CommPkg::communicate_T(A, has_vals);
        }

        // Vector Communication        
        aligned_vector<double>& communicate(ParVector& v,
                const int block_size = 1, const int vblock_offset = 0)
        {
            return CommPkg::communicate(v, block_size);
        }

        void init_comm(ParVector& v, const int block_size = 1, const int vblock_size = 1)
        {
            CommPkg::init_comm(v, block_size, vblock_size);
        }

        // Helper Methods
        aligned_vector<double>& get_double_buffer()
        {
            return buffer;
        }
        aligned_vector<int>& get_int_buffer()
        {
            return int_buffer;
        }

        // Class Attributes
        int recv_size;
        ParComm* local_S_par_comm;
        ParComm* local_R_par_comm;
        ParComm* local_L_par_comm;
        ParComm* global_par_comm;
    };
}
#endif
