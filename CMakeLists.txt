cmake_minimum_required(VERSION 3.15...3.26)
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

# Base compiler options (always used)
add_compile_options(-fopenmp -std=c++17 -std=gnu++17 -march=native)

# Check AVX-512 features individually and accumulate supported flags
set(AVX512_FLAGS "")

check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
if(COMPILER_SUPPORTS_AVX512F)
    message(STATUS "Compiler supports AVX-512F")
    list(APPEND AVX512_FLAGS "-mavx512f")
    
    # Only check additional AVX-512 features if base AVX-512F is supported
    check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
    if(COMPILER_SUPPORTS_AVX512DQ)
        message(STATUS "Compiler supports AVX-512DQ")
        list(APPEND AVX512_FLAGS "-mavx512dq")
    endif()
    
    check_cxx_compiler_flag("-mavx512vl" COMPILER_SUPPORTS_AVX512VL)
    if(COMPILER_SUPPORTS_AVX512VL)
        message(STATUS "Compiler supports AVX-512VL")
        list(APPEND AVX512_FLAGS "-mavx512vl")
    endif()
    
    # check_cxx_compiler_flag("-mavx512bw" COMPILER_SUPPORTS_AVX512BW)
    # if(COMPILER_SUPPORTS_AVX512BW)
    #     message(STATUS "Compiler supports AVX-512BW")
    #     list(APPEND AVX512_FLAGS "-mavx512bw")
    # endif()
    
    # If we have any AVX-512 support, enable the feature macro
    if(AVX512_FLAGS)
        add_definitions(-DHAVE_AVX512)
        add_compile_options(${AVX512_FLAGS})
        message(STATUS "Enabling AVX-512 flags: ${AVX512_FLAGS}")
    endif()
else()
    message(STATUS "Compiler does NOT support AVX-512")
endif()

set(SOURCES skmap/src/skmap_bindings.cpp skmap/src/ParArray.cpp skmap/src/io/IoArray.cpp skmap/src/transform/TransArray.cpp)
pybind11_add_module(skmap_bindings ${SOURCES})

target_link_libraries(skmap_bindings PRIVATE ${GDAL_LIBRARIES} OpenMP::OpenMP_CXX)