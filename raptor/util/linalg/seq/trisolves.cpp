// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include <vector>

using namespace raptor;

/**************************************************************
*****   Forward Substitution (Ly = b)
**************************************************************
***** Solves for the vector y by forward substitution 
***** (assumes L is lower-triangular and non-singular), and
***** returns the result in vector y.
*****
***** Parameters
***** -------------
***** y : U*
*****    Array in which to place the solution
***** b : T*
*****    Array containing vector data 
**************************************************************/
void COOMatrix::fwd_sub_fanin(Vector& y, Vector& b)
{    
    printf("Forward Substitution Not Implemented for these matrix types\n");
}

void CSRMatrix::fwd_sub_fanin(Vector& y, Vector& b)
{   
    int start, end;
    y.copy(b);
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j=start; j<end-1; j++)
        {
            y.values[i] -= vals[j] * y.values[idx2[j]];
        }
        y.values[i] /= vals[end-1];
	//printf("%f\n", y.values[i]);
	//y.print();
    }
}

void CSCMatrix::fwd_sub_fanin(Vector& y, Vector& b)
{    
    printf("Forward Substitution Not Implemented for these matrix types\n");
}

void COOMatrix::fwd_sub_fanout(Vector& y, Vector& b)
{    
    printf("Forward Substitution Not Implemented for these matrix types\n");
}

// UPDATE THIS TO HAVE THE ALGORITHM FROM TEST_MAT.PY
void CSRMatrix::fwd_sub_fanout(Vector& y, Vector& b)
{   
    printf("Forward Substitution Not Implemented for these matrix types\n");
}

void CSCMatrix::fwd_sub_fanout(Vector& y, Vector& b)
{    
    printf("Forward Substitution Not Implemented for these matrix types\n");
}
/**************************************************************
*****   Backward Substitution (Ux = y)
**************************************************************
***** Solves for the vector x by backward substitution
***** (assumes U is upper-triangular and non-singular), and
***** returns the result in vector x
*****
***** Parameters
***** -------------
***** x : U*
*****    Array in which to place the solution
***** b : T*
*****    Array containing vector data
**************************************************************
void COOMatrix::back_sub(Vector& x, Vector& b)
{    
    printf("Backward Substitution Not Implemented for these matrix types\n");
}
void CSRMatrix::back_sub(Vector& x, Vector& b)
{    
    printf("Backward Substitution Not Implemented for these matrix types\n");
}
void CSCMatrix::back_sub(Vector& x, Vector& b)
{    
    printf("Backward Substitution Not Implemented for these matrix types\n");
}*/
