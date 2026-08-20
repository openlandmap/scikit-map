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
