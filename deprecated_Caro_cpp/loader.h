#ifndef CARO_CPP_LOADER_H
#define CARO_CPP_LOADER_H

#include <iostream>
#include <fstream>
#include "tree.h"
#include "caro.h"


void save_data_point(std::ofstream& outfile, const TreeNode* node, const Caro& board);

#endif //CARO_CPP_LOADER_H
