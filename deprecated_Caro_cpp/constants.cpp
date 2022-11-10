#include "constants.h"

namespace constants {
    extern const double C = sqrt(2);    // EXPLORATION PARAMETER

    extern const int MAX_DIM {30};
    extern const std::string CHAR[3] {"O", ".", "X"};
    extern const std::string *CHAR_P {&CHAR[1]}; // this is to be able to use negative index e.g. CHAR_P[-1] = "O" and CHAR_P[1] = "X"
}
