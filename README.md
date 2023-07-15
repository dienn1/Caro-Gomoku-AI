# Caro/Gomoku AI
This repository is the implementation for my undergraduate thesis at UC Irvine [Monte Carlo Tree Search Augmented With Value Network Trained Through Self-play](https://drive.google.com/file/d/1KzxVA_FaIo3cIZj_xTPBiyM2Khbb2Sao/view)

TODO: Refactor test.py and main.py to receive parameters from a config file

## Prerequisite
PyTorch

pybind11

## C++ backend
This project uses C++ for implementation of Caro and Monte Carlo tree search. A simple convolutional network is also implemented in C++ for inference purpose only, initialized with weight from PyTorch NN (model from pytorch does not act nicely with multiprocessing).

Build the folder Caro_pybind to have the necessary binary for MCTS_pybind, Caro_pybind, and Model_pybind. You might have to change a few line in Caro_pybind/Caro_pybind/CMakeLists.txt to have the build system detect your Python version.

Further instructions can be found on pybind11 guide: [First steps - pybind11 documentation](https://pybind11.readthedocs.io/en/stable/basics.html) and [Build systems - pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html)

## Run the model
The main training loop is implemented in main.py, simply run the file with the paramters set. Checkpoints are saved at PATH+SUB_PATH. Information on the parameters can be found in the thesis. DO NOT change dim=7 and count=5, this is simply to initialize a 7x7 Caro game with 5-in-a-row as the win condition.

Tests and evaluations are done in test.py.
