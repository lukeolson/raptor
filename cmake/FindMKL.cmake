# from https://cmake.org/Wiki/CMake:How_To_Find_Libraries

find_package(PkgConfig)
pkg_check_modules(MKL QUIET mkl)

find_path(MKL_INCLUDE_DIR mkl.h
        HINTS ${PC_MKL_INCLUDEDIR} ${PC_MKL_INCLUDE_DIRS} 
              ${MKL_DIR}/include)
find_library(MKL_SEQ_LIBRARY NAMES mkl_sequential
        HINTS ${PC_MKL_LIBDIR} ${PC_MKL_LIBRARY_DIRS} 
        ${MKL_DIR}/lib ${MKL_DIR}/lib/intel64_lin)
find_library(MKL_CORE_LIBRARY NAMES mkl_core
        HINTS ${PC_MKL_LIBDIR} ${PC_MKL_LIBRARY_DIRS} 
        ${MKL_DIR}/lib ${MKL_DIR}/lib/intel64_lin)
find_library(MKL_ILP_LIBRARY NAMES mkl_intel_lp64
        HINTS ${PC_MKL_LIBDIR} ${PC_MKL_LIBRARY_DIRS} 
        ${MKL_DIR}/lib ${MKL_DIR}/lib/intel64_lin)

set(MKL_LIBRARIES ${MKL_SEQ_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_ILP_LIBRARY} )
set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR} )
set(MKL_COMPILE_FLAGS "-m64")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Mkl  DEFAULT_MSG
        MKL_SEQ_LIBRARY MKL_CORE_LIBRARY MKL_INCLUDE_DIR)

mark_as_advanced(MKL_INCLUDE_DIR MKL_SEQ_LIBRARY MKL_CORE_LIBRARY )
