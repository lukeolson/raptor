#include "core/types.hpp"
#include "core/vector.hpp"
#include "core/par_vector.hpp"
#include "core/par_matrix.hpp"

#include "assert.h"

using namespace raptor;

void ParCSRMatrix::fwd_sub(ParVector& y, ParVector& b)
{
    // Check that communication package has been initialized
    if (comm == NULL){
        comm = new ParComm(partition, off_proc_column_map);
    }

    int rank, num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Set the values of y equal to the values of b
    y.copy(b);

    int i, j, start_off, end_off, start_on, end_on;
    // If rank==0, perform seq fwd sub
    if (rank == 0){

        // Call seq fwd_sub method
        on_proc->fwd_sub(y.local, b.local);

        // Send local updated portion of y to next rank
        MPI_Send(&(y.local)[0], y->local_n, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD);

    }
    // Else receive updated x from previous procs then perform seq fwd sub
    else{

        int recv_cnt = y->local_n * rank;
        Vector off_y(recv_cnt);

        // Receive updated portion of y from previous rank
        MPI_Recv(&(off_y)[0], recv_cnt, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (i=0; i<local_num_rows; i++){
            // Start seq fwd sub on off_proc updating local y
            start_off = off_proc->idx1[i];
            end_off = off_proc->idx1[i+1];
            for (j=start_off; j<end_off; j++){
                y.local[i] -= vals[j] * off_y.values[idx2[j]];
            }
          
            // Perform seq fwd sub on on_proc with updated local y
            start_on = on_proc->idx1[i];
            end_on = on_proc->idx1[i+1];
            for (j=start_on; j<end_on; j++){
                y.local[i] -= vals[j] * y.local[idx2[j]];
            }
            y.local[i] /= vals[end_on-1];
        }

        // Only send your updated y portion if you're not the last process
        if (rank != num_procs-1){

            int send_cnt = recv_count + y->local_n;
            // Appending local y calculations onto off_y calculations
            off_y.values.insert(off_y.values.end(), y.local.begin(), y.local.end());
            // Send updated portion of y to next rank
            MPI_Send(&(off_y.values)[0], send_cnt, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD);

        }
    }
   
}
