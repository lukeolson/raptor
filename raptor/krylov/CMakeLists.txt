# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (WITH_MPI)
    set(par_krylov_SOURCES
        krylov/par_cg.hpp
        krylov/par_cg.cpp
        )
else()
    set(par_krylov_SOURCES
        )
endif()
 
# Create a variable called linalg_SOURCES containing all .cpp files:
set(krylov_SOURCES 
    krylov/cg.hpp
    krylov/cg.cpp
    ${par_krylov_SOURCES}
PARENT_SCOPE)




