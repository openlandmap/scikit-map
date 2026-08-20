cmake_minimum_required(VERSION 3.15...3.26)
project(skmap_bindings LANGUAGES CXX)

option(TESTS "Build unit tests" OFF)

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

# If setup.py detected a system GDAL version (via gdal-config), verify that the
# C++ libgdal found here matches it. A mismatch between the C++ libgdal and the
# Python GDAL/rasterio wheels causes ABI-level runtime crashes.
if(DEFINED SKMAP_EXPECTED_GDAL_VERSION AND NOT SKMAP_EXPECTED_GDAL_VERSION STREQUAL "")
    if(NOT SKMAP_EXPECTED_GDAL_VERSION STREQUAL GDAL_VERSION)
        message(FATAL_ERROR
            "GDAL version mismatch: setup.py detected ${SKMAP_EXPECTED_GDAL_VERSION} "
            "but CMake found ${GDAL_VERSION}. Install a matching libgdal-dev or "
            "rebuild the Python GDAL/rasterio wheels.")
    endif()
    message(STATUS "GDAL version check passed: ${GDAL_VERSION}")
endif()

include(CheckCXXCompilerFlag)

# List of AVX-512 flags to test
set(AVX512_CANDIDATES avx512f avx512dq avx512cd avx512bw avx512vl)
set(AVX512_FLAGS "")

foreach(FLAG IN LISTS AVX512_CANDIDATES)
    set(CXX_FLAG "-m${FLAG}")
    set(VAR_NAME "COMPILER_SUPPORTS_${FLAG}")

    # Check if compiler supports this flag
    check_cxx_compiler_flag("${CXX_FLAG}" ${VAR_NAME})

    # Check CPU support individually via /proc/cpuinfo
    execute_process(
        COMMAND bash -c "grep -m1 -o ${FLAG} /proc/cpuinfo || true"
        OUTPUT_VARIABLE CPU_HAS_FLAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Only append if both compiler and CPU support it
    if(${VAR_NAME} AND CPU_HAS_FLAG)
        list(APPEND AVX512_FLAGS "${CXX_FLAG}")
        message(STATUS "Enabling AVX-512 flag: ${CXX_FLAG}")
    else()
        message(STATUS "Skipping ${CXX_FLAG} (compiler or CPU does not support it)")
    endif()
endforeach()

# Add all supported flags at once
if(AVX512_FLAGS)
    # Always append -mfma when using AVX-512
    list(APPEND AVX512_FLAGS -mfma)
    add_compile_options(${AVX512_FLAGS})
    message(STATUS "Final AVX-512 compile options: ${AVX512_FLAGS}")
endif()

# -------------------------
# LIBRARIES AND EXECS
# -------------------------

# create a core library for tests and bindings

# all sources EXCEPT bindings
set(
    SOURCES
    skmap/src/ParArray.cpp skmap/src/io/IoArray.cpp
    skmap/src/transform/TransArray.cpp
)

add_library(skmap STATIC ${SOURCES})
# we need to tell the static library that it will become part of a shared library in the bindings
set_target_properties(skmap PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(skmap PUBLIC
    ${eigen_SOURCE_DIR}
    ${GDAL_INCLUDE_DIRS}
    skmap/include
    skmap/src
)
target_link_libraries(skmap PUBLIC ${GDAL_LIBRARIES} OpenMP::OpenMP_CXX)
if(AVX512_FLAGS)
    target_compile_options(skmap PUBLIC ${AVX512_FLAGS})
endif()

# -------------------------
# Python bindings
# -------------------------

set(BINDINGS_SOURCES skmap/src/skmap_bindings.cpp)
pybind11_add_module(skmap_bindings ${BINDINGS_SOURCES})
target_link_libraries(skmap_bindings PRIVATE skmap)

# -------------------------
# Tests
# -------------------------
if(TESTS)
    enable_testing()
    add_subdirectory(tests/src)
endif()
