#include <iostream>
#include <math.h>
#include "core/matrix.hpp"
#include "mmio.h"
#include "PCG_ILU.hpp"
#include "test_ILU.hpp"
#include <omp.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>
#include "../examples/timer.hpp"
#include <chrono>
#include "systimer.h"
#include "../examples/clear_cache.hpp"

#define MIN(a,b)           ((a)<(b)?(a):(b))

int main (int argc, char *argv[]) 
{

	int numt, num_sweeps;
	char * filename;

	if (argc > 2){ 
		//numt = atoi(argv[1]);
		num_sweeps = atoi(argv[1]);
		filename = argv[2];
	}

	if(argc <=2){
		std::cout<< "Input number of threads, number of sweeps and matrix file name!" << std::endl;
		exit(-1);
	}
	//std::cout << filename << std::endl;

	//const char * filename = "thermal2.mtx";	
	//const char * filename = "UFL_sample_mat.mm";
	//const char * filename = "block_test.txt";

	//timing variables
	double tm1, tm2, tdiff =0.0;
	int num_tests = 1;

	// Variables to clear cache between tests
    //int cache_len = 10000;
    //double* cache_array = new double[cache_len];

	COOMatrix * mat = read_matrix(filename);
	//COOMatrix *mat = read_matrix2(filename);
    
	/*
	//////////For testing small matrix
	COOMatrix * mat = new COOMatrix(4, 4);

	mat->add_value(0, 0, 11.0);
	mat->add_value(0, 1, 10.0);
	mat->add_value(0, 2, 4.0);
	mat->add_value(0, 3, 12.0);

	mat->add_value(1, 0, 10.0);
	mat->add_value(1, 1, 28.0);
	mat->add_value(1, 2, 2.0);
	mat->add_value(1, 3, 18.0);

	mat->add_value(2, 0, 4.0);
	mat->add_value(2, 1, 2.0);
	mat->add_value(2, 2, 9.0);
	mat->add_value(2, 3, 5.0);

	mat->add_value(3, 0, 12.0);
	mat->add_value(3, 1, 18.0);
	mat->add_value(3, 2, 5.0);
	mat->add_value(3, 3, 31.0);
	*/

	//Change A to CSR
	CSRMatrix* Acsr = new CSRMatrix(mat);

	//Sort matrix by row
	Acsr->sort();


	//Diagonal scaling of the entries	
	diagonal_scaling_csr_symmetric(Acsr);
	
	//uncomment this if you want to run regular icc
	/*
	CSCMatrix* Acsc = new CSCMatrix(Acsr);
	//printf("A csr after scaling \n");
	Acsc->sort();
	
	get_ctime(tm1);
  
	icc(Acsc);

	get_ctime(tm2);

	tdiff += (tm2-tm1);

	tdiff /= num_tests;


	//std::cout << "ICC fine grained clock get time took " << tdiff << std::endl;
    printf ("Total time for ICC was %f seconds.\n", tdiff);
    printf("%f \n",tdiff);

	return 0;*/

	//Convert matrix to COO to get initial guesses for L and U 
	COOMatrix* Acoo = new COOMatrix(Acsr);

	//Initial guess for U factor
	//COOMatrix* L = new COOMatrix(mat->n_rows,mat->n_cols);
	COOMatrix* U = new COOMatrix(mat->n_rows,mat->n_cols);

	initial_guess_U(Acoo, U);	

	
	
	//Convert U to CSC 
	//CSRMatrix* Lcsr = new CSRMatrix(L);
	CSCMatrix* Ucsc = new CSCMatrix(U);
	
	//Sort U
	//Lcsr->sort();
	Ucsc->sort();

	//printf("Initial guess U csc\n");
	//Ucsc->print();
	
	//Tell Amanda that when matrix is in CSC, diagonal element is not the first in each column
	//sort in COOMatrix has bug, deleting A[0][1] , 2nd element in matrix
	
	//Extract attributes for A in CSR
	std::vector<int> & A_rowptr = Acsr->row_ptr();
	std::vector<int> & A_indices = Acsr->cols();
	std::vector<double> & A_data = Acsr->data();
	

	//get arrays corresponding to each matrix
	
	std::vector<int> & U_colptr_bef = Ucsc->col_ptr();
	std::vector<int> & U_indices_bef = Ucsc->rows();
	std::vector<double> & U_data_bef = Ucsc->data();

	std::vector<double> aij_U;
 
    // A loop to copy elements of old vector into new vector
    for (int i=0; i<U_data_bef.size(); i++)
        aij_U.push_back(U_data_bef[i]);
			
	//tdiff = 0.0;
	
	for (int i = 0; i < num_tests; i++){

	//	get_ctime(tm1);
  
		icc_fine_grained(Ucsc, aij_U, num_sweeps, numt);

		//t = clock() - t;
		//double tm2 = sys_timer();

	//	get_ctime(tm2);

	//	tdiff += (tm2-tm1);
		//clear_cache(cache_len, cache_array);
	}

	//tdiff /= num_tests;


	//std::cout << "ICC fine grained clock get time took " << tdiff << std::endl;
    //printf ("Total time for ICC was %f seconds.\n", tdiff);
    	//printf("%f \n",tdiff);

  	//Get arrays corresponding to U after the ICC factorization is done 
//	std::cout << "U after icc" <<std::endl;
//	Ucsc->print();
	//std::cout << "Done icc " << std::endl;

	std::vector<int> & U_colptr = Ucsc->col_ptr();
	std::vector<int> & U_indices = Ucsc->rows();
	std::vector<double> & U_data = Ucsc->data();

	//Only doing timing for ICC
	//exit(1);
	return 0;	
	/*//testing
	std::cout << "U after ICC " << std::endl;	  
	Ucsc->print();
	//
	//
	std::cout << "Ucsc data after =" << " "; 
	for (auto i: U_data)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
	//Remove this
	//
	*/


	//Initialize true solution
	std::vector<double> x_true(mat->n_rows);
	x_true.assign(x_true.size(), 1.0);
	
	//Get rhs
	std::vector<double> b = matrix_vector_product(A_rowptr, A_indices, A_data, x_true, mat->n_rows);
	
/*
	std::cout << "b =" << " "; 
	for (auto i: b)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	

	approximate_apply_U_inverse(U_colptr, U_indices, U_data, b, mat->n_rows,5);

	std::cout << "new b =" << " "; 
	for (auto i: b)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
*/	

	//Initialize initial guess for PCG	
	std::vector<double> x0(mat->n_rows);
	x0.assign(x0.size(), 0.0);
	
	//Set tolarance and max iterations for PCG	
	double tol = 0.000001;
	int max_iter = 2 * mat->n_rows;
	
  	get_ctime(tm1);
  
	//Call PCG to solve 
	int numIter =	PCG_UU(x0, b, A_rowptr, A_indices, A_data, U_colptr,U_indices, U_data, max_iter, tol, mat->n_rows);
	
	get_ctime(tm2);

	tdiff += (tm2-tm1);
	

	//t = clock() - t;
	//std::chrono::high_resolution_clock::duration diff_pcg = std::chrono::high_resolution_clock::now()-start_time_pcg;
	//std::cout << "PCG converged in " << numIter << " iterations." << std::endl;
  	//std::cout << "PCG with ICC fine grained took " << tdiff << std::endl;
	printf("PCG converged in %d iterations. \n", numIter);
    printf ("Total time for PCG was %f seconds.\n", tdiff);
    
	exit(1);


	//t = clock() - t;
  	//printf ("PCG took %f seconds \n",((float)t)/CLOCKS_PER_SEC);

	
	/*
	//Print out final solution
	std::cout << "x-final =" << " "; 
	for (auto i: x0)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
	*/
	
	
	//Initialize initial guess for CG	
	x0.assign(x0.size(), 0.0);

  	//t = clock();
  	auto start_time_cg = std::chrono::high_resolution_clock::now();
  
	//Call CG to solve 
	CG(x0, b, A_rowptr, A_indices, A_data, max_iter, tol, mat->n_rows);
	
	
	//t = clock() - t;
	std::chrono::high_resolution_clock::duration diff_cg = std::chrono::high_resolution_clock::now()-start_time_cg;

  	std::cout << "CG took " << diff_cg.count() << std::endl;
	
//	t = clock() - t;
  //	printf ("CG took %f seconds \n",((float)t)/CLOCKS_PER_SEC);


	/*
	//Print out final solution
	std::cout << "x-final =" << " "; 
	for (auto i: x0)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	*/


	
//	return 0;
}

COOMatrix* read_matrix(const char * filename){
	FILE * f; 
	int ret_code;
	MM_typecode matcode;
	int M, N, nz;

	if ((f = fopen(filename, "r")) == NULL)  {
		std::cout << "Cannot find " << filename << std::endl;
		exit(1);
	}

	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}

	if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
			mm_is_sparse(matcode) )
	{
		printf("Sorry, this application does not support ");
		printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
		exit(1);
	}

	if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
		exit(1);

   	COOMatrix * mat = new COOMatrix(M, N, ceil((nz*1.0)/M));

	for (int i=0; i<nz; i++)
	{
		index_t ridx, cidx;
		data_t val;
		fscanf(f, "%d %d %lg\n", &ridx, &cidx, &val);
		mat->add_value (ridx-1, cidx-1, val);

		// since matrix is real symmetric, must add the opposite 
		// value as well since they do not add it in the file
		if (ridx != cidx)
			mat->add_value (cidx-1, ridx-1, val);
	}

	if (f !=stdin) fclose(f);
	//std::cout << "Finished reading matrix" << std::endl;
	
	return mat;
}


COOMatrix* read_matrix2(const char * filename){
	FILE * f; 
	int ret_code;
	int M, N, nz, blocks;

	if ((f = fopen(filename, "r")) == NULL)  {
		std::cout << "Cannot find " << filename << std::endl;
		exit(1);
	}

	fscanf(f, "%d %d %d %d\n", &M, &N, &nz, &blocks);

   	COOMatrix * mat = new COOMatrix(M, N, ceil((nz*1.0)/M));

	for (int i=0; i<nz; i++)
	{
		index_t ridx, cidx;
		data_t val;
		fscanf(f, "%d %d %lg\n", &ridx, &cidx, &val);
		mat->add_value (ridx, cidx, val);
	}

	if (f !=stdin) fclose(f);
	//std::cout << "Finished reading matrix" << std::endl;
	
	return mat;
}

void printInMM(COOMatrix * mat){

	std::vector <int> & row_vec = mat->rows();
	std::vector <int> & col_vec = mat->cols();
	std::vector <double> & data_vec = mat->data();


	std::cout << mat->n_rows << " " << mat->n_cols << " " << mat->nnz << std::endl; 
	
	for(int i = 0; i < row_vec.size(); i++){
		std::cout << row_vec[i]+1 << " " << col_vec[i]+1 << " "<< data_vec[i] << std::endl; 
	}


}

void diagonal_scaling(COOMatrix * mat){
	std::vector <int> & row_vec = mat->rows();
	std::vector <int> & col_vec = mat->cols();
	std::vector <double> & data_vec = mat->data();
	
	//diagonal scaling of the entries
	int r = 0;
	
	while(r < mat->nnz){
		 int i = r+1;
		 while(row_vec[i] == row_vec[r]){
			 data_vec[i] /= data_vec[r];
			 i++;
		 }
		 data_vec[r]=1.0;
		 r = i;
	}
	//std::cout << "Finished diagonal scaling" << std::endl;
}

void diagonal_scaling_csr(CSRMatrix * Acsr){
	std::vector<int> & A_rowptr = Acsr->row_ptr();
	std::vector<int> & A_indices = Acsr->cols();
	std::vector<double> & A_data = Acsr->data();

	std::vector<double> diag_vec(Acsr->n_rows);

	//Get vector of diagonal entries
	for (int row = 0; row<Acsr->n_rows; row++){
		int row_start = A_rowptr[row];
		int row_end = A_rowptr[row+1];
		for(int j = row_start; j < row_end; j++){
			int col = A_indices[j];
			if(row == col)
				diag_vec[row] = A_data[j];
		}
	}

	for(int row = 0; row < Acsr->n_rows;row++){
			int row_start = A_rowptr[row];
			int row_end = A_rowptr[row+1];
		
			for(int j = row_start; j < row_end; j++){
				A_data[j] /= diag_vec[row];
			}
	}
	//std::cout << "Finished diagonal scaling" << std::endl;
}

void diagonal_scaling_csr_symmetric(CSRMatrix * Acsr){
	std::vector<int> & A_rowptr = Acsr->row_ptr();
	std::vector<int> & A_indices = Acsr->cols();
	std::vector<double> & A_data = Acsr->data();

	std::vector<double> diag_vec(Acsr->n_rows);
	
	//Assuming the diagonal entry is stored first in each row
	//for (int row = 0; row<Acsr->n_rows; row++){
	//	diag_vec[row] = A_data[A_rowptr[row]];
	//}

	//Get vector of diagonal entries
	for (int row = 0; row<Acsr->n_rows; row++){
		int row_start = A_rowptr[row];
		int row_end = A_rowptr[row+1];
		for(int j = row_start; j < row_end; j++){
			int col = A_indices[j];
			if(row == col)
				diag_vec[row] = A_data[j];
		}
	}
	
	//Scale symmetrically
	for(int row = 0; row < Acsr->n_rows;row++){
			int row_start = A_rowptr[row];
			int row_end = A_rowptr[row+1];
			
			//A_data[row_start] /= diag_vec[row];
			for(int j = row_start; j < row_end; j++){
				int col = A_indices[j];
				//Set diagonal element
				if(row == col){
					A_data[j] /= diag_vec[row];
				}
				else{
					A_data[j] /= sqrt(diag_vec[row]);
					A_data[j] /= sqrt(diag_vec[col]);
				}
			}
	}
	//std::cout << "Finished symmetric diagonal scaling" << std::endl;
}

void initial_guess_U(COOMatrix * mat, COOMatrix * U){
	std::vector <int> & row_vec = mat->rows();
	std::vector <int> & col_vec = mat->cols();
	std::vector <double> & data_vec = mat->data();
	

	for(int i = 0; i < mat->nnz; i++){
		if(row_vec[i] <= col_vec[i]){
			U->add_value(row_vec[i],col_vec[i],data_vec[i]);
		}
	}
}



void initial_guess_L_U(COOMatrix * mat, COOMatrix * L, COOMatrix * U){
	std::vector <int> & row_vec = mat->rows();
	std::vector <int> & col_vec = mat->cols();
	std::vector <double> & data_vec = mat->data();
	

	for(int i = 0; i < mat->nnz; i++){
		if(row_vec[i] > col_vec[i]){
			L->add_value(row_vec[i],col_vec[i],data_vec[i]);
		}
		else{
			U->add_value(row_vec[i],col_vec[i],data_vec[i]);
		}

	}
}

void icc_fine_grained(CSCMatrix* Ucsc, std::vector<double> const & aij_U, int num_sweeps, int numt){
	//get arrays corresponding to each matrix
	std::vector<int> & U_colptr = Ucsc->col_ptr();
	std::vector<int> & U_rowptr = Ucsc->rows();
	std::vector<double> & U_data = Ucsc->data();

	//declare backup U data array
	std::vector<double> U_backup(Ucsc->nnz);

	//Uncomment this out if running with OpenMP!!!!!!!!!!!!
	//omp_set_num_threads(numt);

	double tm1,tm2,tdiff=0.0;
		
/*	//
	std::cout << "Ucsc data =" << " "; 
	for (auto i: U_data)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	std::cout << "aij_U =" << " "; 
	for (auto i: aij_U)
  		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;
*/

	for(int sweep = 0; sweep < num_sweeps; sweep++){
		
		//Backup data from previous sweep
		for(int i = 0; i < Ucsc->nnz; i++){
			U_backup[i] = U_data[i];
		}
		//#pragma omp parallel proc_bind(close)
		#pragma omp parallel
		{
		#pragma omp for nowait
		for(int col_j=0; col_j < Ucsc->n_cols; col_j++){
			//Update U
			int col_j_start = U_colptr[col_j];
			int col_j_end = U_colptr[col_j+1];
			
			for(int row_i_ind=col_j_start; row_i_ind < col_j_end; row_i_ind++){
				int row_i = U_rowptr[row_i_ind];
				double s = aij_U[row_i_ind];
				
				//compute inner product \sum_{k=1}^{j-1} u_ki u_kj
				int col_i_start = U_colptr[row_i];
				int col_i_end = U_colptr[row_i+1];

				int index_Ui = col_i_start;
				int index_Uj = col_j_start;

				while(index_Uj < col_j_end && index_Ui < col_i_end){
					int Ui_row = U_rowptr[index_Ui];
					int Uj_row = U_rowptr[index_Uj];

					if(Ui_row == row_i){
						index_Ui++;
						continue;
					}
					else if(Uj_row == col_j){
						index_Uj++;
						continue;
					}

					if((Ui_row > row_i-1) || (Uj_row > row_i-1)){
						break;
					}

					if(Ui_row == Uj_row){
						s -= U_backup[index_Ui] * U_backup[index_Uj];
						index_Ui++;
						index_Uj++;
					}
					else if(Ui_row < Uj_row){
						index_Ui++;
					}
					else{
						index_Uj++;
					}

				}
				
				//Find diagonal entry
				double diag = 0.0;
				if (row_i != col_j){					
					for(int prev_row_i_ind = col_i_start; prev_row_i_ind < col_i_end; prev_row_i_ind++){
						int prev_row_i=U_rowptr[prev_row_i_ind];
						if(prev_row_i == row_i){
							diag = U_backup[prev_row_i_ind];
							break;
						}	
					}
					//Update U[i,j]		
					U_data[row_i_ind] = s / diag; 				}
				else{
					if(s<0){
						std::cout << "Encountered negative s, cannot take the sqrt" << std::endl;
						exit(1);
					}
					//Update U[i,i]
					U_data[row_i_ind] = std::sqrt(s);
				}
			}

		}
//	}

		get_ctime(tm1);
		for(int i =0; i<1000;i++)		
			#pragma omp flush
		get_ctime(tm2);
		tdiff += (tm2-tm1);
	}
}

		tdiff /= (1000*num_sweeps);
    		printf("time for flush = %f \n",tdiff);


}	

void icc(CSCMatrix* Lcsc){
	//get arrays corresponding to each matrix
	std::vector<int> & l_colptr = Lcsc->col_ptr();
	std::vector<int> & l_rowptr = Lcsc->rows();
	std::vector<double> & l_data = Lcsc->data();
	
	#pragma omp parallel for 	
	for(int col_k=0; col_k < Lcsc->n_cols;col_k++){
		int col_k_start = l_colptr[col_k];
		int col_k_end = l_colptr[col_k+1];

		int diag_index = col_k_start;

		for(int row_i_ind = col_k_start; row_i_ind<col_k_end;row_i_ind++){
			int row_i = l_rowptr[row_i_ind];
			if(row_i == col_k){
				diag_index = row_i_ind;
				l_data[row_i_ind] = std::sqrt(l_data[row_i_ind]);
			}
		}

		for(int row_i_ind = col_k_start; row_i_ind<col_k_end;row_i_ind++){
			int row_i = l_rowptr[row_i_ind];
			if(row_i > col_k){
				l_data[row_i_ind] = l_data[row_i_ind]/l_data[diag_index];
			}
		}

		for(int col_j=col_k+1; col_j<Lcsc->n_cols;col_j++){
			int col_j_start = l_colptr[col_j];
			int col_j_end = l_colptr[col_j+1];
			for(int row_i_index = col_j_start; row_i_index< col_j_end; row_i_index++){
				int row_i = l_rowptr[row_i_index];
				if(row_i > col_k){
					double aik = 0.0;
					double ajk = 0.0;
					
					for(int temp_ind = col_k_start; temp_ind <col_k_end;temp_ind++){
						int temp_row_i = l_rowptr[temp_ind];
						if(temp_row_i==row_i){
							aik = l_data[temp_ind];
						}
						if(temp_row_i==col_j){
							ajk = l_data[temp_ind];
						}
					}

					l_data[row_i_index]=l_data[row_i_index]-aik*ajk;
				}
			}
		}
	}
}

void ilu_fine_grained(CSRMatrix* Lcsr, CSCMatrix* Ucsc, std::vector<double> const & aij_L, std::vector<double> const & aij_U, int num_sweeps){
	//get arrays corresponding to each matrix
	std::vector<int> & L_rowptr = Lcsr->row_ptr();
	std::vector<int> & L_indices = Lcsr->cols();
	std::vector<double> & L_data = Lcsr->data();

	std::vector<int> & U_colptr = Ucsc->col_ptr();
	std::vector<int> & U_indices = Ucsc->rows();
	std::vector<double> & U_data = Ucsc->data();

	//declare backup L and U arrays
	std::vector<double> L_backup(Lcsr->nnz);
	std::vector<double> U_backup(Ucsc->nnz);

	//Uncomment this out if running with OpenMP!!!!!!!!!!!!!!!
	//omp_set_num_threads(2); 

	//Start new ILU algorithm
	for(int sweep = 0; sweep < num_sweeps; sweep++){

		//Backup data from previous sweep	
		for(int i = 0; i < Ucsc->nnz; i++){
			U_backup[i] = U_data[i];
		}
		
		for(int i = 0; i < Lcsr->nnz; i++){
			L_backup[i] = L_data[i];
		}

		//pragma omp parallel for
		for(int row = 0; row < Lcsr->n_rows;row++){
			//Update L

			int row_L_start = L_rowptr[row];
			int row_L_end = L_rowptr[row+1];
		
			for(int j = row_L_start; j < row_L_end; j++){
				int col = L_indices[j];
			
				if(col == row)
					continue;
				
				int row_U_start = U_colptr[col];
				int row_U_end = U_colptr[col+1];
			
				//compute inner product \sum_{k=1}^{j-1} l_{ik} u_{kj}
				int index_U = row_U_start;
				int col_U = (index_U < row_U_end) ? U_indices[index_U]:Ucsc->n_cols;
			
				double sum = 0;

				for(int k = row_L_start ; k <j ; k++){
					int col_L = L_indices[k];

					//find element in U
					while(col_U < col_L){
						index_U++;
						col_U = U_indices[index_U];
					}

					if (col_U == col_L)
						sum += L_data[k]  * U_data[index_U];
				}
			
				//update l_ij
				L_data[j] = (aij_L[j] - sum)/U_data[row_U_start];
				
			}
			//Update U

			int row_U_start = U_colptr[row];
			int row_U_end = U_colptr[row+1];

			for(int j = row_U_start; j < row_U_end; j++){
				int col = U_indices[j];

				row_L_start = L_rowptr[col];
				row_L_end = L_rowptr[col+1];

				//compute \sum_{k=1}^{j-1} l_{ik} u_{kj}
				int index_L = row_L_start;
				int col_L = (index_L < row_L_end)? L_indices[index_L]:Lcsr->n_rows;
				double sum = 0;

				for(int k = row_U_start; k < j; k++){
					int col_U = U_indices[k];

					//find element in L

					while(col_L < col_U){
						index_L++;
						col_L=L_indices[index_L];
					}
					
					if(col_U == col_L)
						sum +=L_data[index_L] * U_data[k];
	
				}

				//update u_ij
				U_data[j] = aij_U[j] - sum;
			}

		}
	}
}

	/*
	//Start new ICC algorithm
	for(int sweep = 0; sweep < num_sweeps; sweep++){
		//pragma omp parallel for

		for (long col = 0; col < Ucsc->n_cols; ++col){
       		unsigned int col_Ui_start = U_colptr[col];
       		unsigned int col_Ui_end   = U_colptr[col + 1];
   
       		for (unsigned int i = col_Ui_start; i < col_Ui_end; ++i){
         		unsigned int row = U_rowptr[i];
   
         		unsigned int col_Uj_start = U_colptr[row];
         		unsigned int col_Uj_end   = U_colptr[row+1];
   
         		// compute \sum_{k=1}^{j-1} u_ki u_kj
         		unsigned int index_Uj = col_Uj_start;
         		unsigned int row_Uj = U_rowptr[index_Uj];
         		s = aij_U[i];
         		for (unsigned int index_Ui = col_Ui_start; index_Ui < i; ++index_Ui){
           			unsigned int row_Ui = U_rowptr[index_Ui];
   
      			     // find element in col j
           			while (row_Uj < row_Ui)	{
             			++index_Uj;
             			row_Uj = U_rowptr[index_Uj];
           			}
   
           			if (row_Uj == row_Ui)
             			s -= U_backup[index_Ui] * U_backup[index_Uj];
         		}
   
         		if (col != row)
           			U_data[i] = s / U_backup[col_Uj_start]; // diagonal element is first in col!
         		else
           			U_data[i] = std::sqrt(s);
       		}
     	}
	}*/





/*
double sparse_inner(int row_i, int col_j,std::vector<int> & U_indices,std::vector<int> & U_colptr, std::vector<double> & U_data,std::vector<int> & L_indices,std::vector<int> & L_rowptr, std::vector<double> & L_data){
	int L_start = L_rowptr[row_i];
	int L_end = L_rowptr[row_i+1]-1;

	int U_start = U_colptr[col_j];
	int U_end = U_colptr[col_j+1]-1;
	
	int L_index = L_start;
	int U_index = U_start;

	double inner_product = 0.0;

	int iteration_count = 0;

	while(U_index <= U_end && L_index <= L_end){
		int L_col = L_indices[L_index];
		int U_row = U_indices[U_index];

		//If it's the diagonal element in U, just move on to the next elements since L is strictly lower triangular

		if(U_row == col_j){
			U_index++;
			continue;
		}

		if((L_col > col_j - 1) || (U_row > col_j - 1)){
			break;
		}

		if (L_col == U_row){
			inner_product += L_data[L_index] * U_data[U_index];
			L_index++;
			U_index++;
		}
		else if(L_col < U_row){
			L_index++;	
		}
		else{
			U_index++;
		}
		iteration_count++;

		if(iteration_count == 100){
			break;
		}
	}
	return inner_product;
}*/






