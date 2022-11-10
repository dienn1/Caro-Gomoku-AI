#ifndef CARO_CPP_CONSTANTS_H
#define CARO_CPP_CONSTANTS_H

#include <cmath>
#include <string>
namespace constants {
    extern const double C;    // EXPLORATION PARAMETER

    extern const int MAX_DIM;
    extern const std::string CHAR[3];
    extern const std::string *CHAR_P; // this is to be able to use negative index e.g. CHAR_P[-1] = "O" and CHAR_P[1] = "X"
}

#endif //CARO_CPP_CONSTANTS_H
