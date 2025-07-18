cmake_minimum_required(VERSION 3.9)
project(skmap_bindings LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG "3.4"
  GIT_SHALLOW TRUE
  GIT_PROGRESS ON
  SYSTEM
)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG "v2.13.6"
  GIT_SHALLOW TRUE
  GIT_PROGRESS ON
  SYSTEM
)

FetchContent_MakeAvailable(eigen pybind11)

#GDAL_INCLUDE_DIRS()
# include_directories("/skmap/lib/python3.12/site-packages/osgeo/")
# find_library()

find_package(GDAL REQUIRED)
find_package(OpenMP REQUIRED)

message(STATUS "GDAL library: ${GDAL_LIBRARIES}")
message(STATUS "GDAL headers: ${GDAL_INCLUDE_DIRS}")

add_subdirectory("${pybind11_SOURCE_DIR}")

include_directories("${eigen_SOURCE_DIR}")
include_directories("${GDAL_INCLUDE_DIRS}")
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

set(SOURCES skmap/src/skmap_bindings.cpp skmap/src/ParArray.cpp skmap/src/io/IoArray.cpp skmap/src/transform/TransArray.cpp)
pybind11_add_module(skmap_bindings ${SOURCES})

target_link_libraries(skmap_bindings PRIVATE ${GDAL_LIBRARIES} OpenMP::OpenMP_CXX)