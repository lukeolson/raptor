// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

#include "mkl.h"

using namespace raptor;

void COOMatrix::mult(std::vector<double>& x, std::vector<double>& b)
{
    for (int i = 0; i < n_rows; i++)
        b[i] = 0.0;
    mult_append(x, b);
}
void COOMatrix::mult_T(std::vector<double>& x, std::vector<double>& b)
{
    for (int i = 0; i < n_cols; i++)
        b[i] = 0.0;

    mult_append_T(x, b);
}
void COOMatrix::mult_append(std::vector<double>& x, std::vector<double>& b)
{ 
    for (int i = 0; i < nnz; i++)
    {
        b[idx1[i]] += vals[i] * x[idx2[i]];
    }
}
void COOMatrix::mult_append_T(std::vector<double>& x, std::vector<double>& b)
{
    for (int i = 0; i < nnz; i++)
    {
        b[idx2[i]] += vals[i] * x[idx1[i]];
    }
}
void COOMatrix::mult_append_neg(std::vector<double>& x, std::vector<double>& b)
{
    for (int i = 0; i < nnz; i++)
    {
        b[idx1[i]] -= vals[i] * x[idx2[i]];
    }
}
void COOMatrix::mult_append_neg_T(std::vector<double>& x, std::vector<double>& b)
{
    for (int i = 0; i < nnz; i++)
    {
        b[idx2[i]] -= vals[i] * x[idx1[i]];
    }
}
void COOMatrix::residual(const std::vector<double>& x, const std::vector<double>& b, 
        std::vector<double>& r)
{
    for (int i = 0; i < n_rows; i++)
        r[i] = b[i];
     
    for (int i = 0; i < nnz; i++)
    {
        r[idx1[i]] -= vals[i] * x[idx2[i]];
    }
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
    r.copy(b);
    mult_neg_append(x, r)
}

void CSCMatrix::mult(std::vector<double>& x, std::vector<double>& b)
{
    for (int i = 0; i < n_rows; i++)
        b[i] = 0.0;
    mult_append(x, b);
}
void CSCMatrix::mult_T(std::vector<double>& x, std::vector<double>& b)
{
    for (int i = 0; i < n_cols; i++)
        b[i] = 0.0;

    mult_append_T(x, b);
}
void CSCMatrix::mult_append(std::vector<double>& x, std::vector<double>& b)
{ 
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[idx2[j]] += vals[j] * x[i];
        }
    }
}
void CSCMatrix::mult_append_T(std::vector<double>& x, std::vector<double>& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[i] += vals[j] * x[idx2[j]];
        }
    }
}
void CSCMatrix::mult_append_neg(std::vector<double>& x, std::vector<double>& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[idx2[j]] -= vals[j] * x[i];
        }
    }
}
void CSCMatrix::mult_append_neg_T(std::vector<double>& x, std::vector<double>& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            b[i] -= vals[j] * x[idx2[j]];
        }
    }
}
void CSCMatrix::residual(const std::vector<double>& x, const std::vector<double>& b, 
        std::vector<double>& r)
{
    for (int i = 0; i < n_rows; i++)
        r[i] = b[i];

    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            r[idx2[j]] -= vals[j] * x[i];
        }
    }
}

