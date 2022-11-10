cmake_minimum_required(VERSION 3.21)
project(Caro_cpp)

set(CMAKE_CXX_STANDARD 17)

#list(APPEND CMAKE_PREFIX_PATH "D:\\PyTorch CPP\\libtorch")
#find_package(Torch REQUIRED)

add_executable(Caro_cpp main.cpp caro.h caro.cpp mcts.h mcts.cpp constants.h constants.cpp loader.cpp loader.h tree.h tree.cpp)
#target_link_libraries(Caro_cpp "${TORCH_LIBRARIES}")
#set_property(TARGET Caro_cpp PROPERTY CXX_STANDARD 17)

#if (MSVC)
#    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#    add_custom_command(TARGET Caro_cpp
#            POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different
#            ${TORCH_DLLS}
#            $<TARGET_FILE_DIR:Caro_cpp>)
#endif (MSVC)
