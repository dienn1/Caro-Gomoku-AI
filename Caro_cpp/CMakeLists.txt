cmake_minimum_required(VERSION 3.23)
project(Caro_cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(Caro_cpp main.cpp caro.h caro.cpp mcts.h mcts.cpp constants.h constants.cpp loader.cpp)
