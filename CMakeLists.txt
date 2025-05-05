cmake_minimum_required(VERSION 3.4...3.18)
include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG "3.4" # https://github.com/ericniebler/range-v3/releases
  GIT_SHALLOW TRUE
  GIT_PROGRESS ON
  SYSTEM
)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG "v2.13" # https://github.com/ericniebler/range-v3/releases
  GIT_SHALLOW TRUE
  GIT_PROGRESS ON
  SYSTEM
)

FetchContent_MakeAvailable(eigen)
FetchContent_MakeAvailable(pybind11)

#find_package(PkgConfig REQUIRED)
#find_package(GDAL CONFIG REQUIRED)
#pkg_check_modules(Eigen3 REQUIRED Eigen)
#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
project(skmap_bindings)

add_subdirectory("${pybind11_SOURCE_DIR}")
include_directories("${eigen_SOURCE_DIR}")

include_directories(skmap/include)
include_directories(skmap/src)

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)

if(COMPILER_SUPPORTS_AVX512F)
        message(STATUS "Compiler supports AVX-512")
        add_definitions(-DHAVE_AVX512)
        add_compile_options(-fopenmp -std=c++17 -std=gnu++17 -march=native -mavx512f -mavx512dq -mavx512vl -mavx512bw)
else()
        add_compile_options(-fopenmp -std=c++17 -std=gnu++17 -march=native)
        message(STATUS "Compiler does NOT support AVX-512")
endif()

add_link_options(-lgomp)
pybind11_add_module(skmap_bindings skmap/src/skmap_bindings.cpp)

target_compile_definitions(skmap_bindings
        PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})