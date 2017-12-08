#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "gallery/matrix_IO.hpp"

using namespace raptor;

void compare(Vector& b, ParVector& b_par)
{
    double b_norm = b.norm(2);
    double b_par_norm = b_par.norm(2);

    assert(fabs(b_norm - b_par_norm) < 1e-06);

    Vector& b_par_lcl = b_par.local;
    for (int i = 0; i < b_par.local_n; i++)
    {
        assert(fabs(b_par_lcl[i] - b[i+b_par.first_local]) < 1e-06);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read in lower triangular matrix
    char* fname = "mesh3em5_low.mtx";
    CSRMatrix* A = readMatrix(fname);
    ParCSRMatrix* A_par = readParMatrix(fname, MPI_COMM_WORLD, 1, 1);

    Vector x(A->n_rows);
    Vector b(A->n_rows);
    b.set_const_value(1.0);

    ParVector x_par(A_par->global_num_cols, A_par->on_proc_num_cols, 
            A_par->partition->first_local_col);
    ParVector b_par(A_par->global_num_rows, A_par->local_num_rows, 
            A_par->partition->first_local_row);
    b_par.set_const_value(1.0);

    A->fwd_sub(x.values, b.values);
    A_par->fwd_sub(x_par, b_par);
    compare(x, x_par);

    // Set x and x_par to same random values
    for (int i = 0; i < x.size(); i++)
    {
        srand(i);
        b[i] = ((double)rand()) / RAND_MAX;
    }
    for (int i = 0; i < b_par.local_n; i++)
    {
        srand(i+b_par.first_local);
        b_par.local[i] = ((double)rand()) / RAND_MAX;
    }
    A->fwd_sub(x, b);
    A_par->fwd_sub(x_par, b_par);
    compare(x, x_par);

    delete A;
    delete A_par;

    MPI_Finalize();
}

