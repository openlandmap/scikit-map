if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build to debug")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Enable CTest
enable_testing()

# --- Fetch GoogleTest only for testing ---
include(FetchContent)

FetchContent_Declare(
  googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# windows stuff
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- Add test executable ---
add_executable(unit_tests
    test_warp_tile.cpp
    test_par_array.cpp
    io/test_io_array.cpp
    transform/test_mangling.cpp
    transform/test_manipulation.cpp
)

target_link_libraries(unit_tests
    GTest::gtest_main
    skmap
)

# --- Register tests with CTest ---
include(GoogleTest)
gtest_discover_tests(unit_tests)
