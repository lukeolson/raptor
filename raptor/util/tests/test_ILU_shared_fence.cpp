#include <iostream>
#include <math.h>
#include "core/matrix.hpp"
#include "mmio.h"
#include "PCG_ILU.hpp"
#include "test_ILU_shared.hpp"
#include <omp.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>
#include "../examples/timer.hpp"
#include <chrono>
#include "systimer.h"
#include "../examples/clear_cache.hpp"

#define MIN(a,b)           ((a)<(b)?(a):(b))

//#ifndef PRINT_VECTOR (a)
//#define PRINT_VECTOR (a) \
//	for (auto i: a) \
///		std::cout << i << ' '; \
//	std::cout << "\n" <<std::endl;
//endif


int main (int argc, char *argv[]) 
{

	//int numt = atoi(argv[1]);
	//int num_sweeps = atoi(argv[2]);

	//set up MPI shared memory
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

	if (provided !=MPI_THREAD_SINGLE) {
		printf("Hey! Something weird happen\n");
		MPI_Finalize();
		exit(-1);
	}


	int myRank, commSize;

	MPI_Comm sm_comm;
	MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED, 0,MPI_INFO_NULL, &sm_comm);
	MPI_Comm_rank(sm_comm,&myRank);
	MPI_Comm_size(sm_comm,&commSize);

	const int root=0;

	int num_sweeps;
	char * filename;

	if (argc > 2 ) {
		num_sweeps = atoi(argv[1]);
		filename = argv[2];
	} // end if //

	if ( (argc <= 2 || num_sweeps < 0 ) ) {  // using rank 0 because of scanf()
		/*if (myRank == 0) {                              // using rank 0 because of scanf()
		  printf("Number of sweeps [1-15]?: ");fflush(stdout);
		  while ( !scanf("%d", &num_sweeps)  || num_sweeps < 0 ) {
		  printf("wrong input value. Try again... ");fflush(stdout);
		  } // end while //
		  } // end if //
		  MPI_Bcast(&num_sweeps, 1, MPI_INT, 0,MPI_COMM_WORLD);*/
		if(myRank == 0){
			printf("Input number of sweeps and the matrix file name!\n");
			MPI_Finalize();
			exit(-1);
		}
	} // endif //


	//const char * filename = "UFL_sample_mat.mm";
	//const char * filename = "block_test.txt";

	//timing variables
	double tm1, tm2, tdiff,elapsed_time;
	int num_tests = 1;

	// Variables to clear cache between tests
	//int cache_len = 10000;
	//double* cache_array = new double[cache_len];

	COOMatrix * mat = read_matrix(filename);
	//COOMatrix *mat = read_matrix2(filename);

	//Change A to CSR
	CSRMatrix* Acsr = new CSRMatrix(mat);

	//Sort matrix by row
	Acsr->sort();

	//Diagonal scaling of the entries	
	diagonal_scaling_csr_symmetric(Acsr);

	//printf("A csr after scaling \n");
	//Acsr->print();

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

	std::vector<int> & U_colptr = Ucsc->col_ptr();
	std::vector<int> & U_indices = Ucsc->rows();
	std::vector<double> & U_data = Ucsc->data();

	std::vector<double> aij_U;

	// A loop to copy elements of old vector into new vector
	for (int i=0; i<U_data.size(); i++)
		aij_U.push_back(U_data[i]);

	//tdiff = 0.0;

	//Need to initialize shared memory arrays
	int * U_colptr_shared;
	int * U_indices_shared;
	double * aij_U_shared;

	double * U_data_shared;
	double * U_backup_shared;
	double * temp_shared;

	//Just added this
	//explicit rank column division
	int size_colptr = U_colptr.size();
	int size_indices = U_indices.size();
	int size_aij = aij_U.size();
	int size_data = U_data.size();

	int mystart, myend;
	int chunksize = size_colptr/commSize;
	mystart = myRank * chunksize;
	myend = MIN((myRank + 1) * chunksize , size_colptr);


	MPI_Win sm_winT0, sm_winT1, sm_winT2, sm_winT3, sm_winT4;
	MPI_Info info;
	MPI_Info_create(&info);
	MPI_Info_set(info, "no_locks", "true");

	if (myRank == root) {
		MPI_Win_allocate_shared((MPI_Aint) size_colptr*sizeof(int),sizeof(int),info,sm_comm,&U_colptr_shared,&sm_winT0);
		MPI_Win_allocate_shared((MPI_Aint) size_indices*sizeof(int),sizeof(int),info,sm_comm,&U_indices_shared,     &sm_winT1);
		MPI_Win_allocate_shared((MPI_Aint) size_aij*sizeof(double),sizeof(double),info,sm_comm,&aij_U_shared,&sm_winT2);
		MPI_Win_allocate_shared((MPI_Aint) size_data*sizeof(double),sizeof(double),info,sm_comm,&U_data_shared,     &sm_winT3);
		MPI_Win_allocate_shared((MPI_Aint) size_data*sizeof(double),sizeof(double),info,sm_comm,&U_backup_shared,&sm_winT4);
	} else {
		MPI_Win_allocate_shared((MPI_Aint) 0,sizeof(int),info,sm_comm,&U_colptr_shared,&sm_winT0);
		MPI_Win_allocate_shared((MPI_Aint) 0,sizeof(int),info,sm_comm,&U_indices_shared,     &sm_winT1);
		MPI_Win_allocate_shared((MPI_Aint) 0,sizeof(double),info,sm_comm,&aij_U_shared,&sm_winT2);
		MPI_Win_allocate_shared((MPI_Aint) 0,sizeof(int),info,sm_comm,&U_data_shared,     &sm_winT3);
		MPI_Win_allocate_shared((MPI_Aint) 0,sizeof(double),info,sm_comm,&U_backup_shared,&sm_winT4);
	} // end if //

	MPI_Info_free(&info);

	MPI_Aint sz;
	int dispUnit;

	MPI_Win_shared_query(sm_winT0, MPI_PROC_NULL, &sz,&dispUnit,&U_colptr_shared);
	MPI_Win_shared_query(sm_winT1, MPI_PROC_NULL, &sz,&dispUnit,&U_indices_shared);
	MPI_Win_shared_query(sm_winT2, MPI_PROC_NULL, &sz,&dispUnit,&aij_U_shared);
	MPI_Win_shared_query(sm_winT3, MPI_PROC_NULL, &sz,&dispUnit,&U_data_shared);
	MPI_Win_shared_query(sm_winT4, MPI_PROC_NULL, &sz,&dispUnit,&U_backup_shared);


	//my timing position
	MPI_Barrier(sm_comm);
	elapsed_time = -MPI_Wtime();

	MPI_Win_fence(MPI_MODE_NOPRECEDE,sm_winT0);
	MPI_Win_fence(MPI_MODE_NOPRECEDE,sm_winT1);
	MPI_Win_fence(MPI_MODE_NOPRECEDE,sm_winT2);
	MPI_Win_fence(MPI_MODE_NOPRECEDE,sm_winT3);
	MPI_Win_fence(MPI_MODE_NOPRECEDE,sm_winT4);


	//initialize the shared vector
	if (myRank == root) {
		for(int i = 0; i < size_colptr; i++){
			U_colptr_shared[i] = U_colptr[i];
		}
		for(int i = 0; i < size_indices; i++){
			U_indices_shared[i] = U_indices[i];
		}
		for(int i = 0; i < size_aij; i++){
			aij_U_shared[i] = aij_U[i];
		}
		for(int i = 0; i < size_data; i++){
			U_data_shared[i] = U_data[i];
		}
		for(int i = 0; i < size_data; i++){
			U_backup_shared[i] = U_data[i];
		}
	} // end if //

	MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED,sm_winT0);
	MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED,sm_winT1);
	MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED,sm_winT2);
	MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED,sm_winT3);
	MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED,sm_winT4);

	//Philipp timing position
	//MPI_Barrier(sm_comm);
	//elapsed_time = -MPI_Wtime();


	//////////////////////////////////////////////////////////

	for(int sweep = 0; sweep < num_sweeps; sweep++){
		//Backup data from previous sweep
		temp_shared = U_backup_shared;
		U_backup_shared = U_data_shared;
		U_data_shared  = temp_shared;

		//call icc
		icc_sweep(U_colptr_shared, U_indices_shared, aij_U_shared, U_data_shared, U_backup_shared, mystart, myend);
		MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED,sm_winT3);

	}	

	//Philipp time stopping position
	//MPI_Barrier(sm_comm);
	//elapsed_time += MPI_Wtime();

	//update the data vector
	for(int i = 0; i < size_data; i++){
		U_data[i] = U_data_shared[i];
	}

	//My time stopping position
	MPI_Barrier(sm_comm);
	elapsed_time += MPI_Wtime();

	if (myRank == root) {
		//printf ("Total time for ICC was %f seconds.\n", elapsed_time);
		printf("%f \n",elapsed_time);
	} // end if //


	MPI_Win_free(&sm_winT0);
	MPI_Win_free(&sm_winT1);
	MPI_Win_free(&sm_winT2);
	MPI_Win_free(&sm_winT3);
	MPI_Win_free(&sm_winT4);

	////////////////////////////////////////////////////////////////////////////////////
	//Only doing timing for ICC
	MPI_Finalize();
	return 0;
	///////////////////////////////////////////////////////////////////////////////////

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

	elapsed_time = 0;
	elapsed_time = -MPI_Wtime();

	//Call PCG to solve 
	int numIter = PCG_UU(x0, b, A_rowptr, A_indices, A_data, U_colptr,U_indices, U_data, max_iter, tol, mat->n_rows);

	elapsed_time += MPI_Wtime();

	if (myRank == root) {
		printf("PCG converged in %d iterations. \n", numIter);
		printf ("Total time for PCG was %f seconds.\n", elapsed_time);
	} // end if //

	/*
	//Print out final solution
	std::cout << "x-final =" << " "; 
	PRINT_VECTOR (x0);
	*/

	//When also doing PCG, uncomment this part/////////////////////////////////////////
	//MPI_Finalize();
	//return 0;
	//////////////////////////////////////////////////////////////////////////////////

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

	for(int row = 0; row < Acsr->n_rows;row++){
		int row_start = A_rowptr[row];
		int row_end = A_rowptr[row+1];

		double diag = A_data[row_start];
		for(int j = row_start; j < row_end; j++){
			A_data[j] = A_data[j]/diag;
		}
	}
	//std::cout << "Finished diagonal scaling" << std::endl;
}

void diagonal_scaling_csr_symmetric(CSRMatrix * Acsr){
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


void icc_fine_grained(CSCMatrix* Ucsc, std::vector<double> const & aij_U, int num_sweeps, int numt,int mystart, int myend){
	//get arrays corresponding to each matrix
	std::vector<int> & U_colptr = Ucsc->col_ptr();
	std::vector<int> & U_rowptr = Ucsc->rows();
	std::vector<double> & U_data = Ucsc->data();

	//declare backup U data array
	std::vector<double> U_backup(Ucsc->nnz);

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
		//Processor 0 backups up fpr now
		//Backup data from previous sweep
		for(int i = 0; i < Ucsc->nnz; i++){
			U_backup[i] = U_data[i];
		}

		//	icc_sweep(U_data, U_backup, U_colptr, U_rowptr, aij_U, mystart, myend);
	}
} // icc_fine_grained //


// icc_sweep //
void icc_sweep(int * U_colptr, int *  U_rowptr, double * aij_U, double * U_data, double * U_backup, int mystart, int myend){
	for(int col_j=mystart; col_j < myend; col_j++){
		int col_j_start = U_colptr[col_j];
		int col_j_end = U_colptr[col_j+1];
		for(int row_i_ind=col_j_start; row_i_ind < col_j_end; row_i_ind++){
			int row_i = U_rowptr[row_i_ind];
			double s = aij_U[row_i_ind];

			//compute \sum_{k=1}^{j-1} u_ki u_kj
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
} // icc_sweep //
