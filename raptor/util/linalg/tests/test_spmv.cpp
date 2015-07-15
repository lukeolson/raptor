#include <gtest/gtest.h>
#include <mpi.h>
#include <math.h>

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "gallery/matrix_IO.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "util/linalg/spmv.hpp"

#define RAPTOR_SPMV_TOL 1e-10
#define RAPTOR_NEAR_ZERO 1e-10

TEST(serialSpmv, basic) {

    using namespace raptor;

    data_t diff;
    index_t size = 3;
    
    Vector x(size), b(size), true_b(size);

    std::vector<Triplet> tripletList;

    /* Creates 3x3 matrix
        1 2 3
        4 5 6
        7 8 9
    */  
    data_t v_ij;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            v_ij = i*size + j + 1;
            tripletList.push_back(Triplet(i, j, v_ij));
        }
    }
    CSR_Matrix A(&tripletList, size, size);

    x << 1,
         1,
         1;

    true_b << 6,
              15,
              24;

    sequentialSPMV(&A, x, &b, 1., 0.);
	
    for (index_t i = 0; i < size; i++)
    {
        diff = fabs( ((data_t)b[i] - true_b[i]) / true_b[i]);
        EXPECT_LT(diff, RAPTOR_SPMV_TOL) << "SpMV error higher "
                                            "than tolerance at " << i;
    }

}

TEST(parallelSpmv, diffusion) 
{

	data_t eps = 1.0;
	data_t theta = 0.0;

	index_t* grid = (index_t*) calloc(2, sizeof(index_t));
	grid[0] = 4;
	grid[1] = 4;

	index_t dim = 2;

	index_t rank, num_procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	data_t* stencil = diffusion_stencil_2d(eps, theta);
	ParMatrix* A = stencil_grid(stencil, grid, dim, CSR);
    delete[] stencil;

	index_t global_num_rows = A->global_rows;
	index_t local_num_rows = A->local_rows;

	// Create the rhs and solution
	ParVector* b = new ParVector(global_num_rows, local_num_rows);
	ParVector* x = new ParVector(global_num_rows, local_num_rows);
    Vector true_b(global_num_rows);

	x->set_const_value(1.);
	b->set_const_value(0.);
	parallel_spmv(A, x, b, 1., 0.);

    // TODO check this somehow (PyAMG?)
    true_b << 5.0/3.0,
              1,
              1,
              5.0/3.0,
              1,
              0,
              0,
              1,
              1,
              0,
              0,
              1,
              5.0/3.0,
              1,
              1,
              5.0/3.0;

    index_t ind;
    data_t answer, diff;
    data_t* data = (b->local)->data();
	for (index_t proc = 0; proc < num_procs; proc++)
	{
		if (proc == rank) {
			for (index_t i = 0; i < local_num_rows; i++)
			{
                ind = i+(A->first_col_diag);
                answer = true_b[ind];
                if (abs(answer) < RAPTOR_NEAR_ZERO) {
                    diff = abs(data[i] - answer);
                } else {
                    diff = abs( (data[i] - answer) / answer);
                }
                EXPECT_LT(diff, RAPTOR_SPMV_TOL) << "SpMV error higher "
                                                    "than tolerance at " << i;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

    delete x;
    delete b;
    delete A;

}

TEST(parallelSpmv, io) 
{
	index_t rank, num_procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char file[] = "LFAT5.mtx";

	ParMatrix* A = readParMatrix(file, MPI_COMM_WORLD, true);
    ASSERT_NE(A, nullptr) << "Error reading matrix!";

	index_t global_num_rows = A->global_rows;
	index_t local_num_rows = A->local_rows;

	// Create the rhs and solution
	ParVector* b = new ParVector(global_num_rows, local_num_rows);
	ParVector* x = new ParVector(global_num_rows, local_num_rows);
    Vector true_b(global_num_rows);

	x->set_const_value(5.);
	parallel_spmv(A, x, b, 1., 0.);

    true_b << -4.594824e2,
               3.1416e7,
               1.52201550,
               3.770112e4,
              -4.477008e2,
               0,
               0,
               0,
               2.35632e1,
               3.1416e7,
               1.5220155,
               3.770112e4,
               4.948272e2,
               6.330456e2;

    index_t ind;
    data_t answer, diff;
    data_t *data = (b->local)->data();
	for (index_t proc = 0; proc < num_procs; proc++)
	{
		if (proc == rank) {
			for (index_t i = 0; i < local_num_rows; i++)
			{
                ind = i+(A->first_col_diag);
                answer = true_b[ind];
                if (abs(answer) < RAPTOR_NEAR_ZERO) {
                    diff = abs(data[i] - answer);
                } else {
                    diff = abs( (data[i] - answer) / answer);
                }
                EXPECT_LT(diff, RAPTOR_SPMV_TOL) << "SpMV error higher "
                                                    "than tolerance at " << i;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

    delete x;
    delete b;
    delete A;
}
