# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
#Create a variable called linalg_SOURCES containing all .cpp files:

if (WITH_PTSCOTCH)
    set(external_linalg_SOURCES
        util/linalg/external/ptscotch.hpp
        )
else()
    set(external_linalg_SOURCES
        ""
        )
endif()

if (WITH_MPI)
    set(par_linalg_SOURCES
        util/linalg/spmv.cpp
        util/linalg/matmult.cpp
        util/linalg/relax.cpp
        util/linalg/repartition.hpp
        util/linalg/repartition.cpp
	util/linalg/trisolves.cpp
        )
else ()
    set (par_linalg_SOURCES 
        ""
        )
endif()

set(linalg_SOURCES 
    util/linalg/seq/matmult.cpp
    util/linalg/seq/relax.cpp
    util/linalg/seq/trisolves.cpp
    ${par_linalg_SOURCES}
    ${external_linalg_SOURCES}
    PARENT_SCOPE
    )




