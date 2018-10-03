#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
using namespace raptor; 

int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
} //end of main() //

TEST(ILUTest, TestsInCore)
{
	int rows[33] = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8};
	int cols[33] = {0, 1, 3, 0, 1, 2, 4, 1 , 2, 5, 0, 3, 4, 6, 1, 3, 4, 5, 7, 2, 4, 5, 8, 3, 6, 7, 4, 6, 7, 8, 5, 7 , 8};
	double vals[33] = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};

	double factors_answer[33] = {4.0, -1.0, -1.0, -0.25, 3.75, -1.0, -1.0, -0.26666667, 3.73333333, -1.0, -0.25, 3.75, -1.0, -1.0, -0.26666667, -0.26666667, 3.46666667, -1.0, -1.0, -0.26785714, -0.28846154, 3.44368132, -1.0, -0.26666667, 3.73333333, -1.0, -0.28846154,-0.26785714, 3.44368132, -1.0, -0.29038692, -0.29038692, 3.41922617};


	COOMatrix* A_coo = new COOMatrix(9, 9);
	
	for(int i = 0; i < 33; i++){
		A_coo->add_value(rows[i], cols[i], vals[i]);	
	}

	CSRMatrix* A = new CSRMatrix(A_coo);
	
	int lof = 0; 

	Matrix* factors = A->ilu_k(lof);

	std::vector<double> factors_data = factors->vals;
	
	for(int i = 0; i < 33; i++){
		ASSERT_NEAR(factors_answer[i], factors_data[i], 1e-06);
	}

	delete A_coo;
	delete A;
	delete factors;

} //end of TEST(ILUTest, TestsInCore)