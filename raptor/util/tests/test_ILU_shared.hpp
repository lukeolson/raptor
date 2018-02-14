#include <iostream>
#include <math.h>
#include "core/matrix.hpp"
#include "PCG_ILU.hpp"

using namespace raptor;

void diagonal_scaling(COOMatrix * mat);

void diagonal_scaling_csr(CSRMatrix * Acsr);

void diagonal_scaling_csr_symmetric(CSRMatrix * Acsr);

void initial_guess_U(COOMatrix * mat, COOMatrix * U);

void icc_fine_grained(CSCMatrix* Ucsc, std::vector<double> const & aij_U, int num_sweeps, int numt);

COOMatrix* read_matrix(const char * filename);

COOMatrix* read_matrix2(const char * filename);

/*
 * This function performs one sweep of the ICC 
 * algorithm.
 *
 * TODO: Change std::vector <T> to T * for MPI
 */
/*
void icc_sweep(std::vector<double> & U_data,    // the data container	
		std::vector<double> & U_backup, 		// backup copy of data
		std::vector<int> & U_colptr,     		// array of column indices
		std::vector<int> & U_rowptr,  			// array of row indices
		std::vector<double> const & aij_U, 		// something
		int mystart, 
		int myend);
*/
void icc_sweep(int * U_colptr, int *  U_rowptr, double * aij_U, double * U_data, double * U_backup, int mystart, int myend);


/*
 * This function performs row diagonal scaling of a COO matrix
 */
void diagonal_scaling(COOMatrix * mat);// matrix in COO format

/*
 * This function performs row diagonal scaling of a CSR matrix
 */
void diagonal_scaling_csr(CSRMatrix * Acsr);//matrix in CSR format

/*
 * This function performs symmetric diagonal row scaling of a CSR matrix
 */
void diagonal_scaling_csr_symmetric(CSRMatrix * Acsr);//matrix in CSR format

/*
 * This function gets the initial guess of the Upper triangular U matrix for the fine grained ICC factorization for a symmetric 
 * matrix U
 */
void initial_guess_U(COOMatrix * mat,//matrix in COO format
					 COOMatrix * U); //Upper triangular matrix in COO format

/*
 * This function performs a fine grained ICC factorization
 */
void icc_fine_grained(CSCMatrix* Ucsc,//matrix U in CSC format
					  std::vector<double> const & aij_U,//data array of the upper triangular part of the matrix A to be factorized 
					  int num_sweeps,//number of sweeps of the fine grained ICC
					  int numt);//number of threads
/*
 * This function performs one sweep of the ICC 
 * algorithm.
 *
 * TODO: Change std::vector <T> to T * for MPI
 */
/*
void icc_sweep(std::vector<double> & U_data,    // the data container	
		std::vector<double> & U_backup, 		// backup copy of data
		std::vector<int> & U_colptr,     		// array of column indices
		std::vector<int> & U_rowptr,  			// array of row indices
		std::vector<double> const & aij_U, 		// something
		int mystart, 
		int myend);
*/
void icc_sweep(int * U_colptr, int *  U_rowptr, double * aij_U, double * U_data, double * U_backup, int mystart, int myend);
