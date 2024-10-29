cmake_minimum_required(VERSION 3.10)
project(skmap_bindings)

include_directories("${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")

add_library(skmap_bindings SHARED
        ${CMAKE_SOURCE_DIR}/skmap_bindings.cpp
        ${CMAKE_SOURCE_DIR}/src/ParArray.cpp
        ${CMAKE_SOURCE_DIR}/src/IoArray.cpp
)
find_package(Python COMPONENTS Interpreter Development)
include_directories(${PYTHON_INCLUDE_DIRS})

target_include_directories(skmap_bindings PUBLIC
        ${PYTHON_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/pybind11/include
        ${CMAKE_SOURCE_DIR}/skmap_bindings
        ${CMAKE_SOURCE_DIR}/skmap_bindings/include
        ${CMAKE_SOURCE_DIR}/skmap_bindings/src
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++17 -std=gnu++17")

target_link_libraries(skmap_bindings
        pybind11
        fftw3
        m
        gomp
        gdal
        eigen3
        ${PYTHON_INCLUDE_DIRS}
)

set_property(TARGET skmap_bindings PROPERTY CXX_STANDARD 17)