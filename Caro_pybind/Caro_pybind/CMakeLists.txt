# CMakeList.txt : CMake project for Caro_pybind, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

include_directories("Eigen")

#set (PYBIND11_PYTHON_VERSION "3.8")
add_subdirectory(pybind11)

pybind11_add_module(Caro_pybind "Caro_pybind.cpp" "caro.h" "caro.cpp" "constants.h" "constants.cpp")
pybind11_add_module(MCTS_pybind "MCTS_pybind.cpp" "caro.h" "caro.cpp" "mcts.h" "mcts.cpp" 
						"constants.h" "constants.cpp" "tree.h" "tree.cpp" "model.h" "model.cpp")
pybind11_add_module(Model_pybind "Model_pybind.cpp" "model.h" "model.cpp" "model_test.h")
