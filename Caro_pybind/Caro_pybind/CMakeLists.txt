# CMakeList.txt : CMake project for Caro_pybind, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(pybind11)
include_directories("Eigen")

pybind11_add_module(Caro_pybind "Caro_pybind.cpp" "caro.h" "caro.cpp" "mcts.h" "mcts.cpp" 
						"constants.h" "constants.cpp" "tree.h" "tree.cpp")
pybind11_add_module(MCTS_pybind "MCTS_pybind.cpp" "caro.h" "caro.cpp" "mcts.h" "mcts.cpp" 
						"constants.h" "constants.cpp" "tree.h" "tree.cpp")
pybind11_add_module(Model_pybind "Model_pybind.cpp" "model.h" "model_test.h")
