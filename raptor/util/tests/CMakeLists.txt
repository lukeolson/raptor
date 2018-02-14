#add_executable(test_spmv_laplacian test_spmv_laplacian.cpp)
#target_link_libraries(test_spmv_laplacian raptor)
#add_test(LaplacianSpMVTest ./test_spmv_laplacian)

#add_executable(test_spmv_aniso test_spmv_aniso.cpp)
#target_link_libraries(test_spmv_aniso raptor)
#add_test(AnisoSpMVTest ./test_spmv_aniso)

#add_executable(test_spmv_random test_spmv_random.cpp)
#target_link_libraries(test_spmv_random raptor)
#add_test(RandomSpMVTest ./test_spmv_random)

#add_executable(test_ILU test_ILU.cpp PCG_ILU.cpp)
#target_link_libraries(test_ILU raptor ${MPI_LIBRARIES})
#add_test(ILUtest ./test_ILU)

#add_executable(test_ilu_shared test_ilu_shared.cpp pcg_ilu.cpp)
#target_link_libraries(test_ilu_shared raptor ${mpi_libraries})
#add_test(ilutestshared ./test_ilu_shared)

#add_executable(test_ILU_shared_fence test_ILU_shared_fence.cpp PCG_ILU.cpp)
#target_link_libraries(test_ILU_shared_fence raptor ${MPI_LIBRARIES})
#add_test(ILUtestsharedfence ./test_ILU_shared_fence)


add_executable(test_diag_scaling test_diag_scaling.cpp)
target_link_libraries(test_diag_scaling raptor)
add_test(diagscal ./test_diag_scaling)



if (WITH_MPI)
    add_executable(test_par_spmv_laplacian test_par_spmv_laplacian.cpp)
    target_link_libraries(test_par_spmv_laplacian raptor)
    add_test(ParLaplacianSpMVTest_2 mpirun -n 2 ./test_par_spmv_laplacian)
    add_test(ParLaplacianSpMVTest_4 mpirun -n 4 ./test_par_spmv_laplacian)
    add_test(ParLaplacianSpMVTest_8 mpirun -n 8 ./test_par_spmv_laplacian)
    add_test(ParLaplacianSpMVTest_16 mpirun -n 16 ./test_par_spmv_laplacian)

    add_executable(test_par_spmv_aniso test_par_spmv_aniso.cpp)
    target_link_libraries(test_par_spmv_aniso raptor)
    add_test(ParAnisoSpMVTest_2 mpirun -n 2 ./test_par_spmv_aniso)
    add_test(ParAnisoSpMVTest_4 mpirun -n 4 ./test_par_spmv_aniso)
    add_test(ParAnisoSpMVTest_8 mpirun -n 8 ./test_par_spmv_aniso)
    add_test(ParAnisoSpMVTest_16 mpirun -n 16 ./test_par_spmv_aniso)

    add_executable(test_par_spmv_random test_par_spmv_random.cpp)
    target_link_libraries(test_par_spmv_random raptor)
    add_test(ParRandomSpMVTest_2 mpirun -n 2 ./test_par_spmv_random)
    add_test(ParRandomSpMVTest_4 mpirun -n 4 ./test_par_spmv_random)
    add_test(ParRandomSpMVTest_8 mpirun -n 8 ./test_par_spmv_random)
    add_test(ParRandomSpMVTest_16 mpirun -n 16 ./test_par_spmv_random)

    #add_executable(test_repartition test_repartition.cpp)
    #target_link_libraries(test_repartition raptor)
endif()
