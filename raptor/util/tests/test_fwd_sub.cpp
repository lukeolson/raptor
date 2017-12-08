#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    // Read in lower triangular matrix
    char* fname = "steam1_low.mtx";
    CSRMatrix* A = readMatrix(fname);

    Vector x(A->n_rows);
    Vector b(A->n_rows);
    b.set_const_value(1.0);

    A->fwd_sub(x, b);
 
    double b_norm = b.norm(2);
    printf("Seq A norm: %f\n", b_norm);

    delete A;
}

