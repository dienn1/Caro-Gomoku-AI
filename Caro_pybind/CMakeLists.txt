# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)

project ("Caro_pybind")

# Include sub-projects.
add_subdirectory ("Caro_pybind")
