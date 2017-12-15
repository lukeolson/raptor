#include "core/types.hpp"
#include "core/vector.hpp"
#include "core/par_vector.hpp"
#include "core/par_matrix.hpp"
#include <vector>

#include "assert.h"

using namespace raptor;

void ParCSRMatrix::fwd_sub(ParVector& y, ParVector& b)
{
    int rank, num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Set the values of y equal to the values of b
    y.copy(b);

    int i, j, start_off, end_off, start_on, end_on;

    //Call sequential algorithm if a single process
    // Else perform fan-in algorithm
    if(num_procs <= 1){
        on_proc->fwd_sub_fanin(y.local, b.local);
    }
    else{
        Vector global_y(global_num_rows); // Global solution values - updated by everyone
	// Holds idx of first non-updated col in on process rows
	std::vector<int> local_on_ptrs(on_proc->n_rows); 
	// Holds idx of first non-updated col in off process rows
	std::vector<int> local_off_ptrs;

	int local_start, local_stop, local_i, root, on_proc_i, global_i, check_indx, global_col;
        local_start = rank * local_num_rows;
	local_stop = local_start + local_num_rows;
	double temp;

	// Populate pointer arrays to hold current index of non-zero non-updated col in idx2 array
        for(local_i=0; local_i<on_proc->n_rows; local_i++){
		if(on_proc->idx1[local_i] < on_proc->idx1[local_i+1]){
		    local_on_ptrs[local_i] = on_proc->idx1[local_i];
		}
		else{ 
		    local_on_ptrs[local_i] = -1;
		}
	}
	if(rank != 0){
	    for(local_i=0; local_i<off_proc->n_rows; local_i++){
		if(off_proc->idx1[local_i] < off_proc->idx1[local_i+1]){
		    local_off_ptrs.push_back(off_proc->idx1[local_i]);
		}
		else{
		    local_off_ptrs.push_back(-1);
		}
	    }
	}

	for (i=0; i<global_num_rows; i++){ 

		root = i / local_num_rows;
		on_proc_i = i % local_num_rows;

		end_on = on_proc->idx1[on_proc_i+1]; // Gives diagonal element for division

		if (local_start <= i && i < local_stop){
			y.local[on_proc_i] /= on_proc->vals[end_on-1];
			temp = y.local[on_proc_i];
		}
		MPI_Bcast(&temp, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
		global_y.values[i] = temp;

		if (rank > root){
			for(local_i=0; local_i<local_num_rows; local_i++){
				end_off = off_proc->idx1[local_i+1];

				check_indx = local_off_ptrs[local_i];
				if(check_indx != -1){
					if(i == off_proc->idx2[check_indx]){
						global_i = rank*local_num_rows + local_i;
						y.local[local_i] -= off_proc->vals[check_indx] * global_y.values[i];
						printf("Updated: [%d][%d], local_i: %d, check_indx: %d\n", global_i, off_proc->idx2[check_indx], local_i, check_indx);
						if (check_indx+1 < end_off){
							local_off_ptrs[local_i]++;
						}
					}
				}
			}
		}
		else{
			for(local_i=on_proc_i+1; local_i<local_num_rows; local_i++){
				end_on = on_proc->idx1[local_i+1];

				check_indx = local_on_ptrs[local_i];
				if( rank != 0){
					global_col = local_num_rows * rank + on_proc->idx2[check_indx];
				}
				else{
					global_col = on_proc->idx2[check_indx];
				}
				if(check_indx != -1){
					if(i == global_col){
						global_i = rank*local_num_rows + local_i;
						y.local[local_i] -= on_proc->vals[check_indx] * global_y.values[i];
						printf("Updated: [%d][%d], local_i: %d\n", global_i, global_col, local_i);
						if (check_indx+1 < end_on){
							local_on_ptrs[local_i]++;
						}
					}
				}
			}
		}
	}
    }
}


/*void ParCSRMatrix::fwd_sub(ParVector& y, ParVector& b)
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
        //on_proc->fwd_sub(y.local, b.local);

	y.local.copy(b.local);
	for(int i=0; i<on_proc->n_rows; i++){
	    start_on = on_proc->idx1[i];
	    end_on = on_proc->idx1[i+1];
	    for(int j=start_on; j<end_on-1; j++){
	        y.local[i] -= on_proc->vals[j] * y.local[on_proc->idx2[j]];	    
	    }
	    y.local[i] /= on_proc->vals[end_on-1];
	}

        // Send local updated portion of y to next rank if there's more than 1 process
	if ( num_procs > 1){
            MPI_Send(&(y.local)[0], y.local_n, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD);
	    printf("Rank %d, sends %d \n", rank, y.local_n);
	}
    }
    // Else receive updated x from previous procs then perform seq fwd sub
    else{

        int recv_cnt = y.local_n * rank;
        Vector off_y(recv_cnt);

	printf("Rank %d, receives %d \n", rank, recv_cnt);

        // Receive updated portion of y from previous rank
        MPI_Recv(&(off_y)[0], recv_cnt, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (i=0; i<local_num_rows; i++){
            // Start seq fwd sub on off_proc updating local y
            start_off = off_proc->idx1[i];
            end_off = off_proc->idx1[i+1];
            for (j=start_off; j<end_off; j++){
                y.local[i] -= off_proc->vals[j] * off_y.values[off_proc->idx2[j]];
            }
          
            // Perform seq fwd sub on on_proc with updated local y
            start_on = on_proc->idx1[i];
            end_on = on_proc->idx1[i+1];
            for (j=start_on; j<end_on-1; j++){
                y.local[i] -= on_proc->vals[j] * y.local[on_proc->idx2[j]];
            }
            y.local[i] /= on_proc->vals[end_on-1];
        }

        // Only send your updated y portion if you're not the last process
        if (rank != num_procs-1){

            int send_cnt = recv_cnt + y.local_n;

            printf("Rank %d sends %d \n", rank, send_cnt);

            // Appending local y calculations onto off_y calculations
            off_y.values.insert(off_y.values.end(), y.local.values.begin(), y.local.values.end());
            // Send updated portion of y to next rank
            MPI_Send(&(off_y.values)[0], send_cnt, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD);

        }
    }
   
}*/
