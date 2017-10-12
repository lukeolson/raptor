#include "PCG_ILU.hpp"
#include "core/matrix.hpp"

using namespace raptor;


/*
// Finite difference mat with no preconditioner
int main(int argc, char*argv[])
{
	int dim = 5;
	static const double arr1[] = {0.0, 0.0, 0.0, 0.0, 1.0};
	std::vector<double> b (arr1, arr1 + sizeof(arr1) / sizeof(arr1[0]) );

	static const int arr2[] = {0, 2, 5, 8, 11, 13};
	std::vector<int> A_indptr (arr2, arr2 + sizeof(arr2) / sizeof(arr2[0]) );

	static const int arr3[] = {0, 1, 1, 0, 2, 2, 1, 3, 3, 2, 4, 4, 3};
	std::vector<int> A_indices (arr3, arr3 + sizeof(arr3) / sizeof(arr3[0]) );

	std::vector<double> A_data(13);
	A_data[0] = 2.0; A_data[1] = -1.0; A_data[2] = 2.0; A_data[3] = -1.0; A_data[4] = -1.0; A_data[5] = 2.0;
	A_data[6] = -1.0; A_data[7] = -1.0; A_data[8] = 2.0; A_data[9] = -1.0; A_data[10] = -1.0;
	A_data[11] = 2.0; A_data[12] = -1.0;
	
	// use no preconditioner, i.e. L = U = I
	std::vector<int> L_indptr(dim + 1);  // indptr is all zeros (diagonal isn't stored)

	static const int arr5[] = {0, 1, 2, 3, 4, 5};
	std::vector<int> U_indptr (arr5, arr5 + sizeof(arr5) / sizeof(arr5[0]) );

	static const int arr6[] = {0, 1, 2, 3, 4};
	std::vector<int> U_indices (arr6, arr6 + sizeof(arr6) / sizeof(arr6[0]) );


	std::vector<double> U_data(5);
	for (int i = 0; i < 5; i++) U_data[i] = 1.0;



	std::vector<int> L_indices = U_indices; //can be anything
	std::vector<double> L_data = U_data; //can be anything

	std::vector<double> x(dim);  //use initial guess of zero

	double tol = 0.1;
	int max_iter = 5000;

	PCG(x, b, A_indptr, A_indices, A_data, L_indptr, L_indices, L_data, U_indptr, U_indices,U_data,
			max_iter, tol, dim);

	for (int i = 0; i < dim; i++)
		std::cout << x[i] << std::endl;



}*/



void apply_L_inverse(std::vector<int> const & L_indptr, std::vector<int> const & L_indices, std::vector<double> const & L_data, std::vector<double> & x, int dim)
{
	// L is CSR format
	// unit diagonal is not stored in L
	
	int start, end, i, j;
	for (i = 0; i < dim; i ++){
		start = L_indptr[i];
		end = L_indptr[i+1];
		for (j = start; j < end; j++)
			x[i] -= L_data[j] * x[L_indices[j]];
	
	}



}

void apply_U_inverse(std::vector<int> const & U_indptr, std::vector<int> const & U_indices, std::vector<double> const & U_data, std::vector<double> & x, int dim)
{	// U is CSC format
	// first element in a column is on diagonal
	int start, end, i, j;
	for (i = dim - 1; i > -1; i--){
		start = U_indptr[i];
		end = U_indptr[i+1];
		x[i] /= U_data[start];
		for (j = start + 1; j < end; j++)
			x[U_indices[j]] -= U_data[j] * x[i];
	
	}


}

void apply_U_T_inverse(std::vector<int> const & U_indptr, std::vector<int> const & U_indices, std::vector<double> const & U_data, std::vector<double> & x, int dim)
{
	// U is CSC format
	// first element in a column is on diagonal
	// BUT solve lower triangular system U^T x = b
	//
	int start, end, i, j;
	for (i = 0; i < dim; i++){
		start = U_indptr[i];
		end = U_indptr[i+1];
		for (j = start + 1; j<end; j++){
			x[i] -= U_data[j] * x[U_indices[j]];	
		}
		x[i] /= U_data[start];
	
	}

}

void approximate_apply_U_inverse(std::vector<int> const & U_indptr, std::vector<int> const & U_indices, std::vector<double> const & U_data, std::vector<double> & z, int dim, int numiter)
{	
	//Solve Rx = z
	
	//create a copy of U data
	std::vector<double> U_data_copy;
 
    for (int i=0; i<U_data.size(); i++)
         U_data_copy.push_back(U_data[i]);

	//create a copy of z 
	std::vector<double> z_copy;
 
    for (int i=0; i<z.size(); i++)
         z_copy.push_back(z[i]);


/*	
	//Testing
	std::cout<< "U before row scaling: " << std::endl;
	std::cout << "U data =" << " "; 
	for (auto i: U_data_copy)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
	
	std::cout << "U colptr =" << " "; 
	for (auto i: U_indptr)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	std::cout << "U rows =" << " "; 
	for (auto i: U_indices)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
*/	


	//diagonal row scaling of U 
	for(int col = 0; col < dim; col++){
		int col_start = U_indptr[col];
		int col_end = U_indptr[col+1];
		for(int j = col_start; j < col_end; j++){
			int row = U_indices[j];
			double diag = U_data[U_indptr[row]];
			U_data_copy[j] = -U_data_copy[j]/diag;

		}
	}
	
/*	
	//Testing
	std::cout<< "U after row scaling: " << std::endl;
	std::cout << "U data =" << " "; 
	for (auto i: U_data_copy)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
	
	std::cout << "U colptr =" << " "; 
	for (auto i: U_indptr)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	std::cout << "U rows =" << " "; 
	for (auto i: U_indices)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
	
*/
	//zero out the diagonal of U
	for(int col = 0; col < dim; col++){
		int col_start = U_indptr[col];
		U_data_copy[col_start] = 0.0;
	}
	
/*	
	std::cout<< "U after zeroing out the diagonal: " << std::endl;
	std::cout << "U data =" << " "; 
	for (auto i: U_data_copy)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
	
	std::cout << "U colptr =" << " "; 
	for (auto i: U_indptr)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	std::cout << "U rows =" << " "; 
	for (auto i: U_indices)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
*/	

	//scale z
	for(int i = 0; i < z.size(); i++){
		z_copy[i] = z_copy[i]/ U_data[U_indptr[i]];
	}
	
/*	
	std::cout << "b after scaling =" << " "; 
	for (auto i: z_copy)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
*/	

	
	//create vector x 
	std::vector<double> x;
 
    for (int i=0; i<z.size(); i++)
         x.push_back(0.0);



	//Jacobi Iteration
	for(int iter = 0; iter < numiter; iter++){
		 //std::cout<< "i = " << iter << std::endl;
		 x =saxpy(z_copy, U_vector_product(U_indptr, U_indices,U_data_copy,x,dim), 1.0, dim);
		 
		 //std::cout << " x =" << " "; 
		 //for (auto i: x)
  		//	std::cout << i << ' ';
		// std::cout << "\n" <<std::endl;
		 

	}

	for(int i =0; i < z.size();i++){
		z[i] = x[i];
	}
	
/*	
	std::cout << "final x =" << " "; 
	for (auto i: z)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
*/	

}






void approximate_apply_U_T_inverse(std::vector<int> const & U_indptr, std::vector<int> const & U_indices, std::vector<double> const & U_data, std::vector<double> & z, int dim, int numiter)
{	
	//Solve R^T x = z
	
	//create a copy of U data
	std::vector<double> U_data_copy;
 
    for (int i=0; i<U_data.size(); i++)
         U_data_copy.push_back(U_data[i]);

	//create a copy of z 
	std::vector<double> z_copy;
 
    for (int i=0; i<z.size(); i++)
         z_copy.push_back(z[i]);
/*
	//Testing
	std::cout<< "U before row scaling: " << std::endl;
	std::cout << "U data =" << " "; 
	for (auto i: U_data_copy)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
	
	std::cout << "U colptr =" << " "; 
	for (auto i: U_indptr)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	std::cout << "U rows =" << " "; 
	for (auto i: U_indices)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

*/

	//diagonal row scaling of U 
	for(int col = 0; col < dim; col++){
		int col_start = U_indptr[col];
		int col_end = U_indptr[col+1];
		for(int j = col_start; j < col_end; j++){
			int row = U_indices[j];
			double diag = U_data[U_indptr[col]];
			U_data_copy[j] = -U_data_copy[j]/diag;

		}
	}
/*
	//Testing
	std::cout<< "U after row scaling: " << std::endl;
	std::cout << "U data =" << " "; 
	for (auto i: U_data_copy)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
	
	std::cout << "U colptr =" << " "; 
	for (auto i: U_indptr)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	std::cout << "U rows =" << " "; 
	for (auto i: U_indices)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
*/

	//zero out the diagonal of U
	for(int col = 0; col < dim; col++){
		int col_start = U_indptr[col];
		U_data_copy[col_start] = 0.0;
	}
/*
	std::cout<< "U after zeroing out the diagonal: " << std::endl;
	std::cout << "U data =" << " "; 
	for (auto i: U_data_copy)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
	
	std::cout << "U colptr =" << " "; 
	for (auto i: U_indptr)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	std::cout << "U rows =" << " "; 
	for (auto i: U_indices)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
*/

	//scale z
	for(int i = 0; i < z.size(); i++){
		z_copy[i] = z_copy[i]/ U_data[U_indptr[i]];
	}
/*
	std::cout << "b after scaling =" << " "; 
	for (auto i: z_copy)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
*/

	
	//create vector x 
	std::vector<double> x;
 
    for (int i=0; i<z.size(); i++)
         x.push_back(0.0);



	//Jacobi Iteration
	for(int iter = 0; iter < numiter; iter++){
		 //std::cout<< "i = " << iter << std::endl;
		 x =saxpy(z_copy, U_T_vector_product(U_indptr, U_indices,U_data_copy,x,dim), 1.0, dim);
		 //std::cout << " x =" << " "; 
		 //for (auto i: x)
  		//	std::cout << i << ' ';
		// std::cout << "\n" <<std::endl;

	}

	for(int i =0; i < z.size();i++){
		z[i] = x[i];
	}
/*
	std::cout << "final x =" << " "; 
	for (auto i: z)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
*/


}



// satisfied that L and U inverse both work...next matvec?

std::vector<double> matrix_vector_product(std::vector<int> const & A_indptr, std::vector<int> const & A_indices, std::vector<double> const & A_data, std::vector<double> const & x, int dim){
	int start, end, i, j;
	double temp;
	std::vector<double> out(dim);

	for (i = 0; i < dim; i++){
		temp = 0.0;
		start = A_indptr[i];
		end = A_indptr[i+1];
		for (j = start; j < end; j++)
			temp += A_data[j]*x[A_indices[j]];

		out[i] = temp;

	}

	return out;

}

std::vector<double> L_vector_product(std::vector<int> const & L_indptr, std::vector<int> const & L_indices, std::vector<double> const & L_data, std::vector<double> const & x, int dim){
	int start, end, i, j;
	double temp;
	std::vector<double> out(dim);

	for (i = 0; i < dim; i++){
		temp = 0.0;
		start = L_indptr[i];
		end = L_indptr[i+1];
		for (j = start; j < end; j++)
			temp += L_data[j]*x[L_indices[j]];

		out[i] = temp + x[i];

	}

	return out;

}

std::vector<double> U_vector_product(std::vector<int> const & U_indptr, std::vector<int> const & U_indices, std::vector<double> const & U_data, std::vector<double> const & x, int dim){
	int start, end, i, j;
	double temp;
	std::vector<double> out(dim);

	for (j = 0; j < dim; j++){
		start = U_indptr[j];
		end = U_indptr[j+1];
		for (i = start; i < end; i++){
			out[U_indices[i]] += U_data[i]*x[j];
		}
	}

	return out;

}

std::vector<double> U_T_vector_product(std::vector<int> const & U_indptr, std::vector<int> const & U_indices, std::vector<double> const & U_data, std::vector<double> const & x, int dim){
	int start, end, i, j;
	double temp;
	std::vector<double> out(dim);

	for (j = 0; j < dim; j++){
		start = U_indptr[j];
		end = U_indptr[j+1];
		for (i = start; i < end; i++){
			out[j] += U_data[i]*x[U_indices[i]];
		}
	}

	return out;

}

double inner_product(std::vector<double> const & x, std::vector<double> const & y, int dim){
	
	double out = 0.0;

	for (int i = 0; i < dim; i++)
		out += x[i]*y[i];

	return out;

}

std::vector<double> saxpy(std::vector<double> const & x, std::vector<double> const & y, double alpha, int dim)
{
	std::vector<double> z(dim);
	for (int i = 0; i < dim; i++){
		z[i] = x[i] + alpha*y[i];
	}
	return z;

}

double vector_2norm(std::vector<double> const & x, int dim){
	double norm = 0.0;

	for (int i = 0; i < dim; i++)
		norm += x[i]*x[i];

	norm = sqrt(norm);
	return norm;

}

void PCG(std::vector<double> & x, std::vector<double> const & b, std::vector<int> const & A_indptr, std::vector<int> const & A_indices, 
		std::vector<double> const & A_data,std::vector<int> const & L_indptr, std::vector<int> const & L_indices, std::vector<double> const & L_data,
		std::vector<int> const & U_indptr, std::vector<int> const & U_indices, std::vector<double> const & U_data, int max_iter, double tol, int dim)
{
	
	std::vector<double> Ax = matrix_vector_product(A_indptr, A_indices,A_data,x,dim);
	std::vector<double> r = saxpy(b, Ax, -1.0, dim);
	std::vector<double> z(r);
	apply_L_inverse(L_indptr, L_indices,L_data,z, dim);
	apply_U_inverse(U_indptr, U_indices,U_data,z, dim);
	std::vector<double> p(z);
	//vector<double> rnew(dim);
	double alpha, beta;
	double r_dot_z, r_dot_zNew = 0.0;

	r_dot_z = inner_product(r, z, dim);
	for (int k = 0; k < max_iter; k++){
		
		Ax = matrix_vector_product(A_indptr, A_indices, A_data,p , dim);

		alpha = r_dot_z / inner_product(p, Ax, dim);
		x = saxpy(x, p, alpha, dim);

		r = saxpy(r,Ax, -alpha,dim);

		if (vector_2norm(r, dim) < tol){
			break;
		}

		z = r;
		apply_L_inverse(L_indptr, L_indices, L_data, z, dim);
		apply_U_inverse(U_indptr, U_indices, U_data, z, dim);

		r_dot_zNew = inner_product(z, r, dim);
		
		beta = r_dot_zNew / r_dot_z;

		p = saxpy(z, p, beta, dim);
		

		// update 'old' values
		r_dot_z = r_dot_zNew;
	}


}

int PCG_UU(std::vector<double> & x, std::vector<double> const & b, std::vector<int> const & A_indptr, std::vector<int> const & A_indices, 
	std::vector<double> const & A_data,std::vector<int> const & U_indptr, std::vector<int> const & U_indices,
	std::vector<double> const & U_data, int max_iter, double tol, int dim)
{
	
	std::vector<double> Ax = matrix_vector_product(A_indptr, A_indices,A_data,x,dim);
	std::vector<double> r = saxpy(b, Ax, -1.0, dim);
	std::vector<double> z(r);

	apply_U_T_inverse(U_indptr, U_indices, U_data, z, dim);
	apply_U_inverse(U_indptr, U_indices,U_data,z, dim);
	std::vector<double> p(z);
	//vector<double> rnew(dim);
	double alpha, beta;
	double r_dot_z, r_dot_zNew = 0.0;

	r_dot_z = inner_product(r, z, dim);

	int k;

	for (k = 0; k < max_iter; k++){
		
		Ax = matrix_vector_product(A_indptr, A_indices, A_data,p , dim);

		alpha = r_dot_z / inner_product(p, Ax, dim);
		x = saxpy(x, p, alpha, dim);

		r = saxpy(r,Ax, -alpha,dim);

		if (vector_2norm(r, dim) < tol){
			//std::cout << "PCG converged in " << k << "iterations" << std::endl;
			break;
		}

		z = r;
		apply_U_T_inverse(U_indptr, U_indices, U_data, z, dim);
		apply_U_inverse(U_indptr, U_indices, U_data, z, dim);

		r_dot_zNew = inner_product(z, r, dim);
		
		beta = r_dot_zNew / r_dot_z;

		p = saxpy(z, p, beta, dim);
		

		// update 'old' values
		r_dot_z = r_dot_zNew;
	}

	//std::cout << "PCG converged in " << k << " iterations" << std::endl;
	return k;

}


void CG(std::vector<double> & x, std::vector<double> const & b, std::vector<int> const & A_indptr, std::vector<int> const & A_indices, 
	std::vector<double> const & A_data, int max_iter, double tol, int dim)
{
	
	std::vector<double> Ax = matrix_vector_product(A_indptr, A_indices,A_data,x,dim);
	std::vector<double> r = saxpy(b, Ax, -1.0, dim);
	std::vector<double> z(r);

	//apply_U_T_inverse(U_indptr, U_indices, U_data, z, dim);
	//apply_U_inverse(U_indptr, U_indices,U_data,z, dim);
	std::vector<double> p(z);
	double alpha, beta;
	double r_dot_z, r_dot_zNew = 0.0;

	r_dot_z = inner_product(r, z, dim);

	int k;

	for (k = 0; k < max_iter; k++){
		
		Ax = matrix_vector_product(A_indptr, A_indices, A_data,p , dim);

		alpha = r_dot_z / inner_product(p, Ax, dim);
		x = saxpy(x, p, alpha, dim);

		r = saxpy(r,Ax, -alpha,dim);

		if (vector_2norm(r, dim) < tol){
			break;
		}

		z = r;
		//apply_U_T_inverse(U_indptr, U_indices, U_data, z, dim);
		//apply_U_inverse(U_indptr, U_indices, U_data, z, dim);

		r_dot_zNew = inner_product(z, r, dim);
		
		beta = r_dot_zNew / r_dot_z;

		p = saxpy(z, p, beta, dim);
		

		// update 'old' values
		r_dot_z = r_dot_zNew;
	}

	std::cout << "CG converged in " << k << " iterations" << std::endl;


}
