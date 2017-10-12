#include <vector>
#include <iostream>
#include <cmath>

/*
 * This function performs a forward substitution on a lower triangular matrix L, stored in CSR format, i.e. computes L^{-1}x
 */
void apply_L_inverse(std::vector<int> const & L_indptr,//array of row pointers of L 
					std::vector<int> const & L_indices,//array of column indices of L 
					std::vector<double> const & L_data,//data array of L 
					std::vector<double> & x,//vector x on which to apply the inverse 
					int dim);//size of the vector x

/*
 * This function performs a backward substitution on an upper triangular matrix U, stored in CSC format, i.e. computes U^{-1}x
 */
void apply_U_inverse(std::vector<int> const & U_indptr,//array of column pointers of U
					 std::vector<int> const & U_indices,//array of row indices of U
					 std::vector<double> const & U_data,//data array of U
					 std::vector<double> & x,//vector x on which to apply the inverse
					 int dim);//size of the vector x

/*
 * This function performs an approximate backward substitution on an upper triangular matrix U, stored in CSC format, i.e. computes * U^{-1}z
 */
void approximate_apply_U_inverse(std::vector<int> const & U_indptr,//array of column pointers of U
								 std::vector<int> const & U_indices,//array of row indices of U
								 std::vector<double> const & U_data,//data array of U
								 std::vector<double> & z,//vector z on which to apply the inverse
								 int dim,//size of the vector z
								 int numiter);//number of iterations for Jacobi

/*
 * This function performs a forward substitution on the lower triangular matrix U^{T}, stored in CSC format, 
 * i.e. computes U^{-T}x
 */
void apply_U_T_inverse(std::vector<int> const & U_indptr,//array of column pointers of U
					   std::vector<int> const & U_indices,//array of row indices of U
					   std::vector<double> const & U_data,//data array of U
					   std::vector<double> & x,//vector x on which to apply the inverse
					   int dim);//size of the vector x

/*
 * This function performs an approximate forward substitution on the lower triangular matrix U^{T}, stored in CSC format, 
 * i.e. computes U^{-T}z
 */
void approximate_apply_U_T_inverse(std::vector<int> const & U_indptr,//array of column pointers of U
								   std::vector<int> const & U_indices,//array of row indices of U
								   std::vector<double> const & U_data,//data array of U
								   std::vector<double> & z,//vector z on which to apply the inverse
								   int dim,//size of the vector z
								   int numiter);//number of iterations for Jacobi

/*
 * This function computes matrix vector product Ax
 */
std::vector<double> matrix_vector_product(std::vector<int> const & A_indptr,//array of ind pointers
										  std::vector<int> const & A_indices,//array of indices
										  std::vector<double> const & A_data,//data array
										  std::vector<double> const & x,//input vector 
										  int dim);//size of vector

/*
 * This function computes the inner product x^{T}y
 */
double inner_product(std::vector<double> const & x,//input vector
					 std::vector<double> const & y,//second input vector
					 int dim);//size of vectors

/*
 * This function computes the SAXPY x+alpha*y 
 */
std::vector<double> saxpy(std::vector<double> const & x,//input vector
						  std::vector<double> const & y,//second input vector
						  double alpha,//mutiple
						  int dim);//size of vectors

/*
 * This function computes the matrix vector product with a lower triangular matrix L, stored in CSR format
 */
std::vector<double> L_vector_product(std::vector<int> const & L_indptr,//array of row pointers 
									 std::vector<int> const & L_indices,//array of column indices
									 std::vector<double> const & L_data,//data array
									 std::vector<double> const & x,//vector
									 int dim);//size of vector

/*
 * This function computes the matrix vector product with an upper triangular matrix U, stored in CSC format
 */
std::vector<double> U_vector_product(std::vector<int> const & U_indptr,//array of column pointers
									 std::vector<int> const & U_indices,//array of row indices
									 std::vector<double> const & U_data,//data array
									 std::vector<double> const & x,//vector
									 int dim);//size of vector

/*
 * This function computes the matrix  vector product with the transpose of an upper triangular matrix U, stored in CSC format
 */
std::vector<double> U_T_vector_product(std::vector<int> const & U_indptr,//array of column pointers
									   std::vector<int> const & U_indices,//array of row indices
									   std::vector<double> const & U_data,//data array
									   std::vector<double> const & x,//vector
									   int dim);//size of vector

/*
 * This function computes the 2-norm of a vector
 */
double vector_2norm(std::vector<double> const & x,//vector
					int dim);//size of vector

/*
 * This function performs the Preconditionned Conjugate Gradient to solve the system Ax = b with preconditionners L, a lower
 * triangular matrix, stored in CSR format, and U, an upper triangular matrix, stored in CSC format
 */
void PCG(std::vector<double> & x,//solution vector
		 std::vector<double> const & b,//right hand side
		 std::vector<int> const & A_indptr,//array of ind pointers
		 std::vector<int> const & A_indices,//array of indices
		 std::vector<double> const & A_data,//data array
		 std::vector<int> const & L_indptr,//array of row pointers
		 std::vector<int> const & L_indices,//array of column indices
		 std::vector<double> const & L_data, //data array
		 std::vector<int> const & U_indptr,//array of column pointers
		 std::vector<int> const & U_indices,//array of row pointers
		 std::vector<double> const & U_data,//data array
		 int max_iter,//maximum number of iterations
		 double tol,//tolerance
		 int dim);//size of the rhs

/*
 * This function performs Preconditionned Conjugate Gradient to solve the system Ax = b with an upper triangular preconditionner 
 * matrix U
 */
int PCG_UU(std::vector<double> & x,//solution vector
		   std::vector<double> const & b,//right hand side
		   std::vector<int> const & A_indptr,//array of ind pointers 
		   std::vector<int> const & A_indices,//array of indices 
		   std::vector<double> const & A_data,//data array
		   std::vector<int> const & U_indptr,//array of column pointers 
		   std::vector<int> const & U_indices,//array of row indices
		   std::vector<double> const & U_data,//data array 
		   int max_iter,//maximum number of iterations 
		   double tol,//tolerance 
		   int dim);//size of the rhs

/*
 * This function performs Conjugate Gradient to solve the system Ax = b
 */
void CG(std::vector<double> & x,//solution vector
		std::vector<double> const & b,//right hand side
		std::vector<int> const & A_indptr,//array of ind pointers
		std::vector<int> const & A_indices,//array of indices 
		std::vector<double> const & A_data,//data array
		int max_iter,//maximum number of iterations
		double tol,//tolerance
		int dim);//size of the rhs

