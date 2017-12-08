// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

#include "mkl.h"

using namespace raptor;

void COOMatrix::mult(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = 1.0;
    double beta = 0.0;
    mkl_dcoomv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx1.data(), idx2.data(), &nnz, 
                    x.data(), &beta, b.data());
}
void COOMatrix::mult_T(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = 1.0;
    double beta = 0.0;
    mkl_dcoomv("T", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx1.data(), idx2.data(), &nnz, 
                    x.data(), &beta, b.data());
}
void COOMatrix::mult_append(std::vector<double>& x, std::vector<double>& b)
{ 
    double alpha = 1.0;
    double beta = 1.0;
    mkl_dcoomv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx1.data(), idx2.data(), &nnz, 
                    x.data(), &beta, b.data());
}
void COOMatrix::mult_append_T(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = 1.0;
    double beta = 1.0;
    mkl_dcoomv("T", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx1.data(), idx2.data(), &nnz, 
                    x.data(), &beta, b.data());
}
void COOMatrix::mult_append_neg(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = -1.0;
    double beta = 1.0;
    mkl_dcoomv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx1.data(), idx2.data(), &nnz, 
                    x.data(), &beta, b.data());
}
void COOMatrix::mult_append_neg_T(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = -1.0;
    double beta = 1.0;
    mkl_dcoomv("T", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx1.data(), idx2.data(), &nnz, 
                    x.data(), &beta, b.data());
}
void COOMatrix::residual(const std::vector<double>& x, const std::vector<double>& b, 
        std::vector<double>& r)
{
    std::copy(b.begin(), b.end(), r.begin());
    double alpha = -1.0;
    double beta = 1.0;
    mkl_dcoomv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx1.data(), idx2.data(), &nnz, 
                    x.data(), &beta, r.data());
}

void CSRMatrix::mult(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = 1.0;
    double beta = 0.0;
    mkl_dcsrmv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}

void CSRMatrix::mult_T(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = 1.0;
    double beta = 0.0;
    mkl_dcsrmv("T", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}
void CSRMatrix::mult_append(std::vector<double>& x, std::vector<double>& b)
{ 
    double alpha = 1.0;
    double beta = 1.0;
    mkl_dcsrmv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}
void CSRMatrix::mult_append_T(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = 1.0;
    double beta = 1.0;
    mkl_dcsrmv("T", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}
void CSRMatrix::mult_append_neg(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = -1.0;
    double beta = 1.0;
    mkl_dcsrmv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}
void CSRMatrix::mult_append_neg_T(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = -1.0;
    double beta = 1.0;
    mkl_dcsrmv("T", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}
void CSRMatrix::residual(const std::vector<double>& x, const std::vector<double>& b, 
        std::vector<double>& r)
{
    std::copy(b.begin(), b.end(), r.begin());
    double alpha = -1.0;
    double beta = 1.0;
    mkl_dcsrmv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, r.data());
}

void CSCMatrix::mult(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = 1.0;
    double beta = 0.0;
    mkl_dcscmv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}

void CSCMatrix::mult_T(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = 1.0;
    double beta = 0.0;
    mkl_dcscmv("T", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}
void CSCMatrix::mult_append(std::vector<double>& x, std::vector<double>& b)
{ 
    double alpha = 1.0;
    double beta = 1.0;
    mkl_dcscmv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}
void CSCMatrix::mult_append_T(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = 1.0;
    double beta = 1.0;
    mkl_dcscmv("T", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}
void CSCMatrix::mult_append_neg(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = -1.0;
    double beta = 1.0;
    mkl_dcscmv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}
void CSCMatrix::mult_append_neg_T(std::vector<double>& x, std::vector<double>& b)
{
    double alpha = -1.0;
    double beta = 1.0;
    mkl_dcscmv("T", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, b.data());
}
void CSCMatrix::residual(const std::vector<double>& x, const std::vector<double>& b, 
        std::vector<double>& r)
{
    std::copy(b.begin(), b.end(), r.begin());
    double alpha = -1.0;
    double beta = 1.0;
    mkl_dcscmv("N", &n_rows, &n_cols, &alpha, "G**C", vals.data(), 
                    idx2.data(), &(idx1[0]), &(idx1[1]),  
                    x.data(), &beta, r.data());
}

